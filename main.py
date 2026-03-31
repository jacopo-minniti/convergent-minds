import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
import pandas as pd
from tqdm import tqdm
import logging
import sys
import convminds as cm
from convminds.models.vae_adapter import VaeBrainAdapter, info_nce_loss
from convminds.nn.losses import TripartiteVAELoss
from convminds.transforms.pca import VoxelPCA
from convminds.transforms.timeseries import TrimTRs
from transformers import GPT2Tokenizer, GPT2Model

# Configure logging (removed datetime for clean dashboard)
logging.basicConfig(level=logging.DEBUG, format='[%(levelname)s] %(name)s: %(message)s', handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger(__name__)

class GPT2Embedder:
    def __init__(self, model_name="gpt2", device="cpu"):
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.model = GPT2Model.from_pretrained(model_name).to(device).eval()
        self.device = device
        
        # We want the token embedding layer specifically for Phase 2 targets
        self.wte = self.model.wte

    @torch.no_grad()
    def embed_with_context(self, context_texts, target_texts):
        """
        Embeds the target_texts given the context_texts. 
        Returns the mean of the hidden states for the tokens in target_text.
        """
        all_embeddings = []
        for ctx, target in zip(context_texts, target_texts):
            # Combined text: "Context Target"
            full_text = (ctx.strip() + " " + target.strip()).strip()
            if not full_text:
                all_embeddings.append(torch.zeros(1, 768).to(self.device))
                continue
                
            inputs = self.tokenizer(full_text, return_tensors="pt").to(self.device)
            # Use the full model to get hidden states (context-aware)
            outputs = self.model(**inputs)
            # (Batch=1, SeqLen, 768)
            hidden_states = outputs.last_hidden_state
            
            # We want the embeddings corresponding to the TARGET words
            # Simple heuristic: take the last few tokens that match the target length
            # or just take the mean of the whole sequence if target is small.
            # Most accurate is to find where context ends, but mean of full sequence 
            # is a robust proxy for 'the state of the story at TR X'.
            target_ids = self.tokenizer(target.strip(), add_special_tokens=False).input_ids
            num_target_tokens = len(target_ids)
            
            if num_target_tokens > 0:
                # Take the last N tokens (the target)
                target_hidden = hidden_states[:, -num_target_tokens:, :]
                mean_target = target_hidden.mean(dim=1)
            else:
                # If target is empty (silence), take the last state of context
                mean_target = hidden_states[:, -1:, :]
                
            all_embeddings.append(mean_target.squeeze(1))
            
        return torch.cat(all_embeddings, dim=0)

class HuthVaeDataset(Dataset):
    """
    Standardized multi-subject Huth dataset for VAE-InfoNCE training.
    
    Implements:
    - Subject-specific PCA (1000)
    - Hemodynamic shift windowing: BOLD[t+1:t+5] (4 frames)
    - Target windowing: Stimulus[t-2:t+1] (3 frames of context)
    """
    def __init__(self, subject_ids: list[str], trim: int = 5, n_components: int = 1000, device="cpu"):
        self.subject_ids = subject_ids
        self.trim = trim
        self.device = device
        self.n_components = n_components
        
        self.all_samples = []
        self.subject_pcas = {}
        
        # We'll populate these from the benchmarks
        # data[subj][story] = (T, V)
        self.bold_data = {}
        # metadata[subj][story] = {tr_times: [], word_intervals: []}
        self.story_metadata = {}
        
        self._load_and_preprocess()

    def _load_and_preprocess(self):
        logger.info(f"Loading and preprocessing Huth data for subjects: {self.subject_ids}")
        for subj in self.subject_ids:
            logger.info(f"Processing subject {subj}...")
            benchmark = cm.benchmarks.HuthBenchmark(subject=subj)
            
            # 1. Load Story-Units
            # huth_benchmark.human_recording_source returns TOKEN_LEVEL (one story per item)
            recording_data = benchmark.human_recording_source.load_recordings(benchmark, selector={"subject": subj})
            stories = recording_data.values # List of (T, V)
            story_names = recording_data.stimulus_ids
            rois = recording_data.metadata.get("rois", {})
            
            # 2. Fit Subject-Specific PCA
            # We stack all stories for the subject to fit PCA on global variance
            all_subj_bold = np.vstack(stories)
            
            # Setup specific cache path for this subject
            cache_dir = cm.cache.convminds_home() / "cache" / "pca"
            cache_path = cache_dir / f"huth_{subj}_pca_{self.n_components}.joblib"
            
            pca = VoxelPCA(n_components=self.n_components, cache_path=cache_path)
            
            # Create a brain tensor for fitting PCA
            # signal: (B=1, T, Voxels)
            brain_for_pca = cm.data.primitives.BrainTensor(
                torch.from_numpy(all_subj_bold).unsqueeze(0).float(), 
                torch.zeros((all_subj_bold.shape[1], 3)), 
                rois
            )
            
            logger.info(f"Solving PCA for {subj} (fitting on {all_subj_bold.shape[0]} TRs)...")
            pca.fit(brain_for_pca)
            self.subject_pcas[subj] = pca
            
            if pca._pca is not None:
                exp_var = np.sum(pca._pca.explained_variance_ratio_)
                logger.info(f"Subject {subj} PCA Total Explained Variance: {exp_var:.4f}")
                logger.info(f"Top 5 Ratio: {pca._pca.explained_variance_ratio_[:5]}")
            
            # 3. Transform Stories using learned PCA
            self.bold_data[subj] = {}
            for i, story_name in enumerate(story_names):
                # Apply PCA and z-score or similar
                # Huth recordings are often already z-scored, but we'll apply PCA projection
                # Wrap (T, V) into BrainTensor for transform
                bt = cm.data.primitives.BrainTensor(torch.from_numpy(stories[i]).unsqueeze(0).float(), torch.zeros(stories[i].shape[1], 3), rois)
                projected = pca(bt).signal.squeeze(0).numpy() # (T, 1000)
                
                # Z-score normalization per run
                mean = projected.mean(axis=0, keepdims=True)
                std = projected.std(axis=0, keepdims=True) + 1e-8
                projected = (projected - mean) / std
                
                self.bold_data[subj][story_name] = projected
                
            # 4. GATHER STIMULUS ALIGNMENT
            # Use metadata to align brain with words
            for record in benchmark.stimuli:
                story_name = record.stimulus_id
                if story_name not in self.bold_data[subj]:
                    continue
                    
                self.story_metadata[story_name] = record.metadata
                actual_trs = self.bold_data[subj][story_name].shape[0]
                
                # ALIGNMENT STRUCTURE:
                # Target TR: x
                # Context TRs: [x-3, x-2, x-1] (Prompt)
                # Target Text: Words in TR x
                # Brain Input: BOLD in [x+1, x+2, x+3, x+4]
                
                # Limits: 
                # Start: x-3 >= 0 => x >= 3
                # End: x+4 < actual_trs => x < actual_trs - 4
                for x in range(3, actual_trs - 4):
                    self.all_samples.append((subj, story_name, x))
        
        logger.info(f"Preprocessing complete. Total samples: {len(self.all_samples)}")
        

    def __len__(self):
        return len(self.all_samples)

    def __getitem__(self, index):
        subj, story, x = self.all_samples[index]
        
        # BRAIN INPUT: 4 TR window after target stimulus [x+1:x+1+4]
        bold_window = self.bold_data[subj][story][x+1:x+5]
        
        metadata = self.story_metadata[story]
        tr_times = metadata["tr_times"]
        word_intervals = metadata["word_intervals"]
        
        # Helper to get words in a TR range
        def get_words(tr_start, tr_end):
            t_start = tr_times[tr_start]
            t_end = tr_times[tr_end+1] if tr_end+1 < len(tr_times) else tr_times[-1] + 2.0
            
            noise_tokens = {"sp", "uh", "um"}
            words = [i["text"].lower().strip() for i in word_intervals 
                     if i["xmin"] >= t_start and i["xmin"] < t_end]
            words = [w for w in words if w not in noise_tokens and len(w) > 0]
            return " ".join(words) if words else ""

        # CONTEXT: TRs [x-3, x-2, x-1]
        context_text = get_words(x-3, x-1)
        
        # TARGET: TR x
        target_text = get_words(x, x)
        
        return {
            "bold": torch.from_numpy(bold_window).float(), # (4, 1000)
            "context": context_text,
            "target": target_text,
            "subject": subj,
            "story": story,
            "tr": x,
            "time_window": (tr_times[x], tr_times[x+1] if x+1 < len(tr_times) else tr_times[x]+2.0)
        }

if __name__ == "__main__":
    import os
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    logger.info("--- Huth VAE Pipeline Environment ---")
    logger.info(f"CONVMINDS_HOME: {os.environ.get('CONVMINDS_HOME', '~/.convminds (default)')}")
    logger.info(f"HF_HOME: {os.environ.get('HF_HOME', '~/.cache/huggingface (default)')}")
    logger.info(f"Using device: {device}")
    logger.info("---------------------------------------")
    
    # Subjects (Starting with 4 for speed, user suggested 8)
    # Ensure datalad download for ds003020 derivative is ready!
    subject_ids = ["S1"] # Fixed: focusing on S1 as requested
    
    # 1. Dataset & Loaders
    dataset = HuthVaeDataset(subject_ids=subject_ids, trim=5, device=device)
    total_len = len(dataset)
    train_len = int(0.9 * total_len)
    test_len = total_len - train_len
    train_set, test_set = Subset(dataset, range(train_len)), Subset(dataset, range(train_len, total_len))
    
    train_loader = DataLoader(train_set, batch_size=256, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_set, batch_size=256, shuffle=False)
    
    # 2. GPT-2 Embedder for Target Latents
    logger.info("Initializing GPT-2 Embedder...")
    embedder = GPT2Embedder(device=device)
    
    # 3. Model & Loss (Tripartite)
    model = VaeBrainAdapter(input_dim=1000, n_frames=4, latent_dim=768).to(device)
    
    # 7. TRAINING LOOP (Phases 1 & 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    criterion = TripartiteVAELoss()
    scheduler = CosineAnnealingLR(optimizer, T_max=15)
    
    results_comparison = {}
    
    # --- EXPERIMENT RUNNER ---
    def run_experiment(exp_name, use_vae_warmup=True):
        logger.info(f"\n{'='*40}")
        logger.info(f"RUNNING EXPERIMENT: {exp_name}")
        logger.info(f"{'='*40}")
        
        model = VaeBrainAdapter(input_dim=1000, n_frames=4, latent_dim=768).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        scheduler = CosineAnnealingLR(optimizer, T_max=20)
        criterion = TripartiteVAELoss()
        
        total_epochs = 20
        for epoch in range(1, total_epochs + 1):
            model.train()
            epoch_losses = {"rec": [], "align": []}
            
            # Model A: 1-10 Recon, 11-20 Align
            # Model B: 1-20 Align
            if use_vae_warmup:
                phase = 1 if epoch <= 10 else 2
            else:
                phase = 2
            
            pbar = tqdm(train_loader, desc=f"[{exp_name}] Ep {epoch:2d}")
            for batch in pbar:
                x = batch["bold"].to(device)
                h_text = embedder.embed_with_context(batch["context"], batch["target"]).to(device)
                outputs = model(x)
                
                if phase == 1:
                    criterion.rec_weight = 1.0
                    criterion.kl_weight = 0.005
                    criterion.align_weight = 0.0
                    metrics = criterion(
                        recon_x=outputs["x_hat"],
                        x=outputs["x_orig"],
                        mu=outputs["mu"],
                        logvar=outputs["logvar"],
                        z=outputs["z"],
                        h_text=h_text
                    )
                    loss = metrics["loss"]
                else:
                    loss = info_nce_loss(outputs["mu"], h_text)
                    metrics = {"rec_loss": torch.tensor(0.0), "align_loss": loss}

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                if phase == 1:
                    epoch_losses["rec"].append(metrics["rec_loss"].item())
                else:
                    epoch_losses["align"].append(metrics["align_loss"].item())
            
            scheduler.step()
            if phase == 1:
                logger.info(f"Ep {epoch:2d} | Ph 1 | Train MSE (Recon): {np.mean(epoch_losses['rec']):.4f}")
            else:
                logger.info(f"Ep {epoch:2d} | Ph 2 | Train InfoNCE (Align): {np.mean(epoch_losses['align']):.4f}")

        stats = run_eval(model, test_loader)
        results_comparison[exp_name] = stats
        return model, stats
    def calculate_nlp_metrics(pred_text, target_text):
        """Calculates BLEU-1, ROUGE-1, and WER for a single pair of texts."""
        def tokenize(t): return t.lower().split()
        p_toks, t_toks = tokenize(pred_text), tokenize(target_text)
        if not t_toks: return {"bleu": 0.0, "rouge": 0.0, "wer": 0.0}
        if not p_toks: return {"bleu": 0.0, "rouge": 0.0, "wer": 1.0}
        
        # 1. ROUGE-1 (Recall)
        matches = len([w for w in p_toks if w in t_toks])
        rouge = matches / len(t_toks)
        
        # 2. BLEU-1 (Precision + Brevity Penalty)
        precision = matches / len(p_toks)
        bp = 1.0 if len(p_toks) >= len(t_toks) else np.exp(1 - len(t_toks)/len(p_toks))
        bleu = precision * bp
        
        # 3. WER (Word Error Rate - Levenshtein distance)
        d = np.zeros((len(t_toks)+1, len(p_toks)+1))
        for i in range(len(t_toks)+1): d[i,0] = i
        for j in range(len(p_toks)+1): d[0,j] = j
        for i in range(1, len(t_toks)+1):
            for j in range(1, len(p_toks)+1):
                if t_toks[i-1] == p_toks[j-1]: d[i,j] = d[i-1,j-1]
                else: d[i,j] = min(d[i-1,j]+1, d[i,j-1]+1, d[i-1,j-1]+1)
        wer = d[len(t_toks), len(p_toks)] / len(t_toks)
        
        return {"bleu": bleu, "rouge": rouge, "wer": min(wer, 1.0)}

    def run_eval(model, loader):
        model.eval()
        results = {"mse_recon": [], "mse_align": [], "cosine": [], 
                   "top10_acc": [], "bleu": [], "rouge": [], "wer": []}
        
        # We collect all batch results to run retrieval-based decoding
        all_mus, all_targets, all_texts = [], [], []
        
        with torch.no_grad():
            for batch in loader:
                x = batch["bold"].to(device)
                h_text = embedder.embed_with_context(batch["context"], batch["target"]).to(device)
                outputs = model(x)
                
                results["mse_recon"].append(torch.mean(torch.square(outputs["x_hat"] - outputs["x_orig"])).item())
                results["mse_align"].append(torch.mean(torch.square(outputs["mu"] - h_text)).item())
                results["cosine"].append(F.cosine_similarity(outputs["mu"], h_text, dim=-1).mean().item())
                
                # Identification Acc
                if outputs["mu"].shape[0] > 1:
                    cos_sim_matrix = F.cosine_similarity(outputs["mu"].unsqueeze(1), h_text.unsqueeze(0), dim=-1)
                    for i in range(cos_sim_matrix.shape[0]):
                        top10 = torch.topk(cos_sim_matrix[i], k=min(10, cos_sim_matrix.shape[0])).indices
                        results["top10_acc"].append(1.0 if i in top10 else 0.0)
                
                all_mus.append(outputs["mu"])
                all_targets.append(h_text)
                all_texts.extend(batch["target"])
        
        # Retrieval-based NLP Decoding
        # For each mu, find the closest target embedding in the bank
        full_mus = torch.cat(all_mus, dim=0)
        full_targets = torch.cat(all_targets, dim=0)
        
        # Cosine similarity matrix between all predicted brain states and all possible text embeddings
        scores = F.cosine_similarity(full_mus.unsqueeze(1), full_targets.unsqueeze(0), dim=-1)
        for i in range(len(all_texts)):
            # Pick the best text match (excluding self distance check, we want generalization)
            best_match_idx = torch.argmax(scores[i]).item()
            best_text = all_texts[best_match_idx]
            ground_truth = all_texts[i]
            
            metrics = calculate_nlp_metrics(best_text, ground_truth)
            results["bleu"].append(metrics["bleu"])
            results["rouge"].append(metrics["rouge"])
            results["wer"].append(metrics["wer"])
            
        return {k: np.mean(v) if v else 0.0 for k, v in results.items()}

    results_comparison = {}

    # --- EXECUTION ---
    # Model A: Sequential (VAE -> Align)
    model_a, stats_a = run_experiment("Model_A (Sequential)", use_vae_warmup=True)
    
    # Model B: Direct (Align only)
    model_b, stats_b = run_experiment("Model_B (Direct Align)", use_vae_warmup=False)

    # Comparison Report
    logger.info("\n" + "="*60)
    logger.info("FINAL ABLATION COMPARISON REPORT")
    logger.info("="*60)
    logger.info(f"{'Metric':<20} | {'Model A (Seq)':<15} | {'Model B (Dir)':<15}")
    logger.info("-" * 60)
    # Filter for interesting metrics
    metrics_to_show = ["mse_align", "cosine", "top10_acc", "bleu", "rouge", "wer"]
    for k in metrics_to_show:
        if k in stats_a:
            val_a = stats_a[k]
            val_b = stats_b[k]
            # Use percentage for certain metrics
            if k in ["top10_acc", "bleu", "rouge"]:
                logger.info(f"{k:<20} | {val_a*100:<14.2f}% | {val_b*100:<14.2f}%")
            else:
                logger.info(f"{k:<20} | {val_a:<15.4f} | {val_b:<15.4f}")
    logger.info("="*60)
    
    # track first batch samples for A
    test_samples = []
    model_a.eval()
    with torch.no_grad():
        for batch in test_loader:
            x = batch["bold"].to(device)
            contexts = batch["context"]
            targets = batch["target"]
            outputs = model_a(x)
            h_text_batch = embedder.embed_with_context(contexts, targets)
            
            for j in range(min(2, len(targets))):
                hat_vals = outputs["mu"][j][:5].detach().cpu().numpy()
                orig_vals = h_text_batch[j, :5].detach().cpu().numpy()
                test_samples.append({
                    "id": f"{batch['subject'][j]} | {batch['story'][j]} | TR {batch['tr'][j]}",
                    "context": contexts[j],
                    "target": targets[j],
                    "w_count": len(targets[j].split()),
                    "time": batch["time_window"][j],
                    "hat_pca": hat_vals,
                    "orig_pca": orig_vals
                })
            break
    
    logger.info("\nSAMPLE TEST INSTANCES (Numerical Decode Analysis):")
    for i, s in enumerate(test_samples):
        logger.info(f"Sample: {s['id']}")
        logger.info(f"  Time Window:  {s['time'][0].item():.1f}s -> {s['time'][1].item():.1f}s")
        logger.info(f"  Context Text: \"{s['context']}\"")
        logger.info(f"  Target Text:  \"{s['target']}\" ({s['w_count']} words)")
        logger.info(f"  Latent (Align-mu, First 5): {s['hat_pca']}")
        logger.info(f"  Target (LLM-WTE, First 5):  {s['orig_pca']}")
        # Calculate local Pearson for these 5
        local_c = np.corrcoef(s['hat_pca'], s['orig_pca'])[0, 1]
        logger.info(f"  Local Component Correlation: {local_c:.4f}")
        logger.info("-" * 30)

    logger.info("\nUniversal Brain-to-LLM Adapter training finalized.")
