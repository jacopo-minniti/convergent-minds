import os
import sys
import torch
from torch.utils.data import DataLoader
import logging
import argparse

import convminds as cm
from convminds.data.benchmarks.huth_alignment import HuthAlignmentDataset
from convminds.models.residual_steer import ResidualSteerLM
from convminds.pipelines.residual_steer import ResidualSteerPipeline

# Configure logging
logging.basicConfig(
    level=logging.INFO, 
    format='[%(levelname)s] %(name)s: %(message)s', 
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Convergent Minds: Phase-based Brain Steering")
    parser.add_argument("--epochs", type=str, default="5,10", help="Comma-separated epochs for Ph1 and Ph2")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate for the adapter")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--subject", type=str, default="S1", help="Subject ID (e.g., S1, S2)")
    parser.add_argument("--llm", type=str, default="gpt2", help="Base LLM ID from HuggingFace")
    parser.add_argument("--layer", type=int, default=6, help="Injection layer (0-indexed)")
    args = parser.parse_args()
    
    # 0. Global Setup
    cm.set_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    phase_epochs = [int(e) for e in args.epochs.split(",")]
    
    # 1. Dataset & DataLoaders
    logger.info(f"Initializing Huth Alignment Dataset for Subject {args.subject}...")
    train_set = HuthAlignmentDataset(subject_ids=[args.subject], split="train")
    test_set = HuthAlignmentDataset(subject_ids=[args.subject], split="test")
    
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)
    
    # 2. Model Initialization
    logger.info(f"Initializing ResidualSteerLM ({args.llm})...")
    model = ResidualSteerLM(llm_id=args.llm, injection_layer=args.layer)
    
    # 3. Pipeline Execution
    pipeline = ResidualSteerPipeline(model=model, lr=args.lr, device=device)
    
    # Training
    if any(e > 0 for e in phase_epochs):
        pipeline.train(train_loader, phase_epochs=phase_epochs)
    
    # Evaluation
    logger.info("Starting Multi-Baseline Evaluation...")
    results = pipeline.evaluate(test_loader, samples_to_show=2)
    
    # 4. Save Artifacts
    save_path = f"brain_steer_{args.llm}_{args.subject}_{args.layer}.pt"
    torch.save(model.adapter.state_dict(), save_path)
    logger.info(f"Steering Adapter weights saved to {save_path}")

if __name__ == "__main__":
    main()