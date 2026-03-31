import torch
import numpy as np
import logging
import sys
import convminds as cm
from torch.utils.data import DataLoader
from tqdm import tqdm
from tests.huth_vae_pipeline import HuthVaeDataset, GPT2Embedder

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(name)s: %(message)s')
logger = logging.getLogger("hrf_diag")

def diagnose_lag():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    subject = "S1"
    
    logger.info(f"--- HRF Lag Diagnostics for {subject} ---")
    
    # Load dataset with standard trim
    dataset = HuthVaeDataset(subject_ids=[subject], trim=20, n_components=100) # Faster with 100 comps
    
    logger.info("Initializing Embedder...")
    embedder = GPT2Embedder(device=device)
    
    # We'll check lags from -5 to +10 TRs
    lags = range(-5, 11)
    results = {}
    
    # 1. Prepare Text Embeddings for a large chunk of samples
    # We'll take 1000 samples for the diagnostic
    num_samples = 1000
    all_h_text = []
    all_bold_centered = [] # BOLD at index t
    
    logger.info(f"Extracting embeddings for {num_samples} samples...")
    for i in tqdm(range(num_samples)):
        sample = dataset[i]
        subj, story, t = dataset.all_samples[i]
        
        # Get text embedding for the context window ending at t
        h = embedder.embed([sample["text"]])
        all_h_text.append(h.squeeze(0).numpy())
        
        # Center of the BOLD window we want to check (raw PCA at t)
        # We allow offset in the loop
        bold_at_t = dataset.bold_data[subj][story][t]
        all_bold_centered.append(bold_at_t)
        
    H = np.vstack(all_h_text) # (N, 768)
    
    # Check correlations at different lags
    for lag in lags:
        corrs = []
        for i in range(num_samples):
            subj, story, t = dataset.all_samples[i]
            
            # Index for BOLD: t + lag
            idx = t + lag
            if idx < 0 or idx >= dataset.bold_data[subj][story].shape[0]:
                continue
                
            bold_val = dataset.bold_data[subj][story][idx] # (100,)
            text_val = all_h_text[i] # (768,)
            
            # Average correlation across PCA components
            # Since both are approximately zero-mean, dot product or corrcoef
            c = np.corrcoef(bold_val[:10], text_val[:10])[0, 1] # Check first few as proxy
            if not np.isnan(c):
                corrs.append(c)
        
        if corrs:
            avg_c = np.mean(corrs)
            results[lag] = avg_c
            logger.info(f"Lag {lag:2d} TRs: Avg Correlation = {avg_c:.4f}")
        else:
            logger.info(f"Lag {lag:2d} TRs: No valid samples")
            
    best_lag = max(results, key=results.get)
    logger.info(f"--- DIAGNOSTIC COMPLETE ---")
    logger.info(f"Best HRF Lag observed: {best_lag} TRs")
    logger.info(f"(Current pipeline uses approximately +1 to +4 TRs lag)")

if __name__ == "__main__":
    diagnose_lag()
