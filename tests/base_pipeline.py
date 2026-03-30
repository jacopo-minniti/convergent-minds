"""
End-to-end example: predict GPT-2 hidden states from Pereira fMRI with MSE loss.

This script wires together the core convminds abstractions:
- Benchmark (PereiraBenchmark)
- HumanSubject + HFArtificialSubject
- BrainDataModule with preprocessing (HRF windowing, z-score, PCA)
- Simple brain encoder model trained with MSE
"""

# TODO:
# 1. test test.py
# 2. inspect different datasets 

# 1. make README less weird
# 2. add all the cool things in readme for the repo
# 3. add versioning
# 4. publish to pip 

import numpy as np
import torch

import convminds as cm
import convminds.nn as cnn
from convminds.metrics import R2
from tqdm import tqdm
import logging
import sys

# Configure logging to handle Slurm redirecting stdout/stderr correctly
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class BrainToGPT2(cnn.Module):
    def __init__(self, *, num_queries: int = 16, llm_dim: int = 768):
        super().__init__()
        self.encoder = cnn.encoders.SpatialAttentionEncoder(
            num_queries=num_queries,
            query_dim=llm_dim,
            use_coords=True,
        )
        self.readout = torch.nn.Linear(llm_dim, llm_dim)

    def forward(self, brain_tensor):
        latents = self.encoder(brain_tensor)
        pooled = latents.mean(dim=1)
        return self.readout(pooled)


if __name__ == "__main__":
    torch.manual_seed(0)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    logger.info("Loading benchmark...")
    benchmark = cm.benchmarks.PereiraBenchmark(alignment_window=4, reduce="mean")
    
    # Selecting the first subject discovered in the preprocessing
    human = cm.subjects.HumanSubject()
    oracle = cm.subjects.HFArtificialSubject("gpt2", layers=[-1], pooling_strategy="mean")

    datamodule = cm.data.BrainDataModule(
        benchmark=benchmark,
        human_subject=human,
        artificial_subject=oracle,
        stateless_transforms=[cm.transforms.HRFWindow(t=1)],
        stateful_transforms=[cm.transforms.ZScore(dim="batch")],
        batch_size=32,
    )
    logger.info("Setting up DataModule (this may include recording activations)...")
    datamodule.setup()
    
    # Log stimulus preview for debugging
    logger.info(f"Stimulus Sample: {benchmark.stimuli[0].stimulus_id} -> {benchmark.stimuli[0].text}")

    # Diagnostic: Check target variance to ensure R2 makes sense
    all_targets = []
    for batch in datamodule.train_dataloader():
        all_targets.append(batch["target_latents"].numpy())
    all_targets = np.concatenate(all_targets, axis=0)
    logger.info(f"Target stats (train): mean={all_targets.mean():.4f}, std={all_targets.std():.4f}, var={all_targets.var():.4f}")

    model = BrainToGPT2(num_queries=16, llm_dim=768).to(device)
    trainer = cm.trainers.GradientTrainer(model=model, loss_fn=torch.nn.MSELoss(), lr=1e-3)

    logger.info("Starting training...")
    trainer.fit(datamodule.train_dataloader(), target_key="target_latents", epochs=10)

    model.eval()
    losses = []
    all_preds = []
    all_targets = []
    logger.info("Starting evaluation...")
    with torch.no_grad():
        for batch in tqdm(datamodule.test_dataloader(), desc="Evaluating"):
            brain_tensor = batch["brain_tensor"].to(device)
            targets = batch["target_latents"].to(device)
            preds = model(brain_tensor)
            loss = torch.nn.functional.mse_loss(preds, targets)
            losses.append(loss.item())
            all_preds.append(preds.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    r2_val = R2(all_preds, all_targets)

    logger.info(f"Test MSE: {sum(losses) / max(1, len(losses)):.4f}")
    logger.info(f"Test R2: {r2_val:.4f}")

