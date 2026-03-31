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

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s [%(levelname)s] %(name)s: %(message)s', handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger(__name__)

class GPT2Embedder:
    def __init__(self, model_name="gpt2", device="cpu"):
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.model = GPT2Model.from_pretrained(model_name).to(device).eval()
        self.device = device
        
        # We want the token embedding layer specifically for Phase 2 targets
        self.wte = self.model.wte

    @torch.no_grad()
    def embed(self, texts):
        # texts is a list of strings
        # We need the mean of token embeddings for each text
        all_embeddings = []
        for t in texts:
            if not t.strip():
                all_embeddings.append(torch.zeros(1, 768).to(self.device))
                continue
                
            inputs = self.tokenizer(t, return_tensors="pt").to(self.device)
            # Extract raw token embeddings
            # (Batch, SeqLen, 768)
            token_embeds = self.wte(inputs.input_ids)
            # Mean pool across tokens
            mean_embed = token_embeds.mean(dim=1) # (1, 768)
            all_embeddings.append(mean_embed)
            
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
                
                # Valid TRs t for windows [t:t+4] and Text at t-4
                # Start: t-4 >= 0 => t >= 4. Also trim=5 => t >= 5.
                # End: t+4 < actual_trs. Also trim=5 => t < actual_trs - 5.
                for t in range(self.trim, actual_trs - self.trim - 4):
                    self.all_samples.append((subj, story_name, t))
        
        logger.info(f"Preprocessing complete. Total samples: {len(self.all_samples)}")
        

    def __len__(self):
        return len(self.all_samples)

    def __getitem__(self, index):
        subj, story, t = self.all_samples[index]
        
        # INPUT fMRI: 4 TR window [t:t+4]
        bold_window = self.bold_data[subj][story][t:t+4]
        
        # TARGET TEXT: Perception at time T-4 mapped to BOLD at T
        target_t = t - 4 
        
        metadata = self.story_metadata[story]
        tr_times = metadata["tr_times"]
        word_intervals = metadata["word_intervals"]
        
        t_start = tr_times[target_t]
        t_end = tr_times[target_t+1] if target_t+1 < len(tr_times) else tr_times[-1] + 2.0
        t_win = (t_start, t_end)

        # Clean tokens to remove silence cues (sp) and non-semantic fillers
        noise_tokens = {"sp", "uh", "um"}
        context_words = [i["text"].lower().strip() for i in word_intervals 
                        if i["xmin"] >= t_start and i["xmin"] < t_end]
        context_words = [w for w in context_words if w not in noise_tokens and len(w) > 0]
        context_text = " ".join(context_words) if context_words else " "
        
        return {
            "bold": torch.from_numpy(bold_window).float(), # (4, 1000)
            "text": context_text,
            "subject": subj,
            "story": story,
            "tr": t,
            "time_window": t_win
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
            
            # Logic: 
            # Model A (use_vae_warmup=True): 1-10 Recon, 11-20 Align
            # Model B (use_vae_warmup=False): 1-20 Align
            if use_vae_warmup:
                phase = 1 if epoch <= 10 else 2
            else:
                phase = 2
            
            pbar = tqdm(train_loader, desc=f"[{exp_name}] Ep {epoch:2d} Ph {phase}")
            for batch in pbar:
                x = batch["bold"].to(device)
                h_text = embedder.embed(batch["text"]).to(device)
                outputs = model(x)
                
                if phase == 1:
                    # Reconstruction Task
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
                    # Alignment Task (using InfoNCE)
                    # We utilize the 'z' or 'mu' for alignment. 'mu' is more stable.
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
            
            # Compact Logging
            if phase == 1:
                logger.info(f"Ep {epoch:2d} | Train MSE (Recon): {np.mean(epoch_losses['rec']):.4f}")
            else:
                logger.info(f"Ep {epoch:2d} | Train InfoNCE (Align): {np.mean(epoch_losses['align']):.4f}")

        # Final Eval
        stats = run_eval(model, test_loader)
        results_comparison[exp_name] = stats
        return model, stats

    def run_eval(model, loader):
        model.eval()
        results = {"mse_recon": [], "mse_align": [], "cosine": [], "corr": [], "top10_acc": []}
        with torch.no_grad():
            for batch in loader:
                x = batch["bold"].to(device)
                h_text = embedder.embed(batch["text"]).to(device)
                outputs = model(x)
                
                results["mse_recon"].append(torch.mean(torch.square(outputs["x_hat"] - outputs["x_orig"])).item())
                results["mse_align"].append(torch.mean(torch.square(outputs["mu"] - h_text)).item())
                results["cosine"].append(F.cosine_similarity(outputs["mu"], h_text, dim=-1).mean().item())
                
                # Correlation
                flat_hat = outputs["x_hat"].cpu().numpy()
                flat_orig = outputs["x_orig"].cpu().numpy()
                results["corr"].append(np.corrcoef(flat_hat.flatten(), flat_orig.flatten())[0, 1])
                
                # Identification Acc
                if outputs["mu"].shape[0] > 1:
                    cos_sim_matrix = F.cosine_similarity(outputs["mu"].unsqueeze(1), h_text.unsqueeze(0), dim=-1) # (B, B)
                    for i in range(cos_sim_matrix.shape[0]):
                        top10 = torch.topk(cos_sim_matrix[i], k=min(10, cos_sim_matrix.shape[0])).indices
                        results["top10_acc"].append(1.0 if i in top10 else 0.0)
        return {k: np.mean(v) if v else 0.0 for k, v in results.items()}

    # --- EXECUTION ---
    # Model A: Sequential (VAE -> Align)
    model_a, stats_a = run_experiment("Model_A (Sequential)", use_vae_warmup=True)
    
    # Model B: Direct (Align only)
    model_b, stats_b = run_experiment("Model_B (Direct Align)", use_vae_warmup=False)

    # Comparison Report
    logger.info("\n" + "="*50)
    logger.info("ABLATION COMPARISON REPORT")
    logger.info("="*50)
    logger.info(f"{'Metric':<20} | {'Model A (Seq)':<15} | {'Model B (Dir)':<15}")
    logger.info("-" * 55)
    for k in stats_a.keys():
        logger.info(f"{k:<20} | {stats_a[k]:<15.4f} | {stats_b[k]:<15.4f}")
    logger.info("="*50)
    
    # track first batch samples for A
    test_samples = []
    model_a.eval()
    with torch.no_grad():
        for batch in test_loader:
            x = batch["bold"].to(device)
            texts = batch["text"]
            outputs = model_a(x)
            for j in range(min(2, len(texts))):
                hat_vals = outputs["mu"][j][:5].detach().cpu().numpy() # Using MU as proxy for alignment state
                orig_vals = embedder.embed([texts[j]])[0, :5].detach().cpu().numpy()
                test_samples.append({
                    "id": f"{batch['subject'][j]} | {batch['story'][j]} | TR {batch['tr'][j]}",
                    "text": texts[j],
                    "w_count": len(texts[j].split()),
                    "time": batch["time_window"][j],
                    "hat_pca": hat_vals,
                    "orig_pca": orig_vals
                })
            break
    
    logger.info("\nSAMPLE TEST INSTANCES (Numerical Decode Analysis):")
    for i, s in enumerate(test_samples):
        logger.info(f"Sample: {s['id']}")
        logger.info(f"  Time Window:  {s['time'][0].item():.1f}s -> {s['time'][1].item():.1f}s")
        logger.info(f"  Context Text: \"{s['text']}\" ({s['w_count']} words)")
        logger.info(f"  Latent (Align-mu, First 5): {s['hat_pca']}")
        logger.info(f"  Target (LLM-WTE, First 5):  {s['orig_pca']}")
        # Calculate local Pearson for these 5
        local_c = np.corrcoef(s['hat_pca'], s['orig_pca'])[0, 1]
        logger.info(f"  Local Component Correlation: {local_c:.4f}")
        logger.info("-" * 30)

    logger.info("\nUniversal Brain-to-LLM Adapter training finalized.")
