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
from convminds.models.vae_adapter import VaeBrainAdapter
from convminds.nn.losses import TripartiteVAELoss
from convminds.transforms.pca import VoxelPCA
from convminds.transforms.timeseries import TrimTRs
from transformers import AutoTokenizer, AutoModelForCausalLM

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s [%(levelname)s] %(name)s: %(message)s', handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger(__name__)

class GPT2Embedder:
    """Extracts mean-pooled last hidden states from GPT-2 small."""
    def __init__(self, model_id="gpt2", device="cuda"):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(model_id).to(device)
        self.model.eval()

    @torch.no_grad()
    def embed(self, texts: list[str]) -> torch.Tensor:
        encoded = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True).to(self.device)
        output = self.model(**encoded, output_hidden_states=True)
        # pooled: mean across time dimension of the last hidden state
        pooled = output.hidden_states[-1].mean(dim=1)
        return pooled.cpu()

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
                
                # Valid TRs t for windows [t-2:t+1] and BOLD [t+1:t+5]
                # Start: t-2 >= 0 => t >= 2. Also trim=5 => t >= 5.
                # End: t+1 < actual_trs and t+5 < actual_trs => t+4 < actual_trs. Also trim=5 => t < actual_trs - 5.
                for t in range(self.trim, actual_trs - self.trim - 4):
                    self.all_samples.append((subj, story_name, t))
        
        logger.info(f"Preprocessing complete. Total samples: {len(self.all_samples)}")

    def __len__(self):
        return len(self.all_samples)

    def __getitem__(self, index):
        subj, story, t = self.all_samples[index]
        
        # BOLD: [t+1, t+2, t+3, t+4] -> (4, 1000)
        bold_window = self.bold_data[subj][story][t+1:t+5]
        
        # Safety check: ensure uniform length (should never happen now)
        if bold_window.shape[0] != 4:
            # Fallback to absolute index 0 if something is wrong (better than crash)
            return self.__getitem__(0)
        
        # TEXT context: [t-2, t-1, t] (Approx 6s)
        # Using the intervals in metadata
        metadata = self.story_metadata[story]
        tr_times = metadata["tr_times"]
        word_intervals = metadata["word_intervals"]
        
        # Define window boundaries [t_start, t_end]
        # t-2 starts at tr_times[t-2], t ends at tr_times[t+1]? 
        # Actually Huth TRs are midpoint centered? 
        # Let's say we take words between tr_times[t-2] and tr_times[t+1]
        t_start = tr_times[t-2]
        t_end = tr_times[t+1] if t+1 < len(tr_times) else tr_times[-1] + 2.0
        
        context_words = [i["text"] for i in word_intervals if i["xmin"] >= t_start and i["xmin"] < t_end]
        context_text = " ".join(context_words) if context_words else " "
        
        return {
            "bold": torch.from_numpy(bold_window).float(), # (4, 1000)
            "text": context_text,
            "subject": subj,
            "story": story,
            "tr": t
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
    loss_fn = TripartiteVAELoss(rec_weight=1.0, kl_weight=0.005, align_weight=0.5, temperature=0.07).to(device)
    
    # 4. Optimization
    optimizer = AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    scheduler = CosineAnnealingLR(optimizer, T_max=100) # Simplified scheduler for brevity
    
    logger.info("Starting Training Loop...")
    for epoch in range(1, 11): # 10 epochs for demo
        model.train()
        epoch_losses = []
        for batch in tqdm(train_loader, desc=f"Epoch {epoch}", leave=False):
            # x: BOLD features (batch, 4, 1000)
            x = batch["bold"].to(device)
            texts = batch["text"]
            
            # a) Get Text Targets
            h_text = embedder.embed(texts).to(device) # (B, 768)
            
            # b) VAE Forward
            outputs = model(x)
            
            # c) Loss
            metrics = loss_fn(
                recon_x=outputs["x_hat"], 
                x=outputs["x_orig"], 
                mu=outputs["mu"], 
                logvar=outputs["logvar"],
                z=outputs["z"],
                h_text=h_text
            )
            
            optimizer.zero_grad()
            metrics["loss"].backward()
            optimizer.step()
            epoch_losses.append(metrics["loss"].item())
            
        avg_loss = sum(epoch_losses) / len(epoch_losses)
        logger.info(f"Epoch {epoch} complete. Avg Loss: {avg_loss:.4f}")
        scheduler.step()
        
    logger.info("Universal Brain-to-LLM Adapter training finalized.")
