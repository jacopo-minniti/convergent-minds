import os
import sys
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
from tqdm import tqdm
import logging
import argparse

# convminds imports
from convminds.data.benchmarks.huth_alignment import HuthAlignmentDataset
from convminds.nn.encoders.language import GPT2Embedder
from convminds.models.brain_adapters import BrainLanguageAdapter
from convminds.nn.losses import info_nce_loss, vae_reconstruction_loss
from convminds.nn.metrics import calculate_nlp_metrics, identification_accuracy

# Configure logging
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(name)s: %(message)s', handlers=[logging.StreamHandler(sys.stdout)])
logging.getLogger("absl").setLevel(logging.ERROR)
logger = logging.getLogger(__name__)

def run_eval(model, loader, embedder, device):
    model.eval()
    stats = {"mse_align": [], "cosine": [], "top10_acc": [], "bleu": [], "rouge1": [], "rougeL": [], "wer": [], "meteor": []}
    
    all_mus, all_targets, all_texts = [], [], []
    
    with torch.no_grad():
        for batch in loader:
            x, context, target = batch["bold"].to(device), batch["context"], batch["target"]
            h_text = embedder.embed_with_context(context, target).to(device)
            outputs = model(x)
            
            # Use h_llm for alignment stats (this is mu in VAE mode)
            h_pred = outputs["h_llm"]
            
            stats["mse_align"].append(F.mse_loss(h_pred, h_text).item())
            stats["cosine"].append(F.cosine_similarity(h_pred, h_text, dim=-1).mean().item())
            stats["top10_acc"].append(identification_accuracy(h_pred, h_text, top_k=10))
            
            all_mus.append(h_pred)
            all_targets.append(h_text)
            all_texts.extend(target)
            
    # LLM Retrieval Dashboard (Memory-Efficient MatMul)
    full_mus = torch.cat(all_mus, dim=0)
    full_targets = torch.cat(all_targets, dim=0)
    
    # Normalize for cosine similarity via dot product
    mus_norm = F.normalize(full_mus, p=2, dim=-1)
    targets_norm = F.normalize(full_targets, p=2, dim=-1)
    scores = torch.matmul(mus_norm, targets_norm.t()) # (N, N)
    
    for i in range(len(all_texts)):
        best_idx = torch.argmax(scores[i]).item()
        nlp = calculate_nlp_metrics(all_texts[best_idx], all_texts[i])
        for k, v in nlp.items():
            stats[k].append(v)
            
    return {k: np.mean(v) for k, v in stats.items()}

def main():
    parser = argparse.ArgumentParser(description="Convergent Minds: Huth Brain-to-LLM Alignment")
    parser.add_argument("--use-vae", action="store_true", help="Use a VAE stem (reconstruction + KL) instead of a simple MLP.")
    parser.add_argument("--loss-type", type=str, choices=["info_nce", "mse"], default="info_nce", help="Loss function for alignment.")
    parser.add_argument("--epochs", type=str, default="20", help="Comma-separated phase epochs: '10,10' for VAE warmup then Align, or '20' for joint.")
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Devices: {device} | Use VAE: {args.use_vae} | Loss: {args.loss_type}")

    # 1. Load Data
    logger.info("Initializing Huth Alignment Benchmark...")
    train_set = HuthAlignmentDataset(subject_ids=["S1"], split="train")
    test_set = HuthAlignmentDataset(subject_ids=["S1"], split="test")
    
    train_loader = DataLoader(train_set, batch_size=256, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_set, batch_size=256, shuffle=False)
    
    # 2. Setup Models
    embedder = GPT2Embedder(device=device)
    model = BrainLanguageAdapter(use_vae=args.use_vae).to(device)
    optimizer = AdamW(model.parameters(), lr=1e-4)
    
    # 3. Model Saving Path (Move up to check cache)
    save_dir = os.path.join(os.environ.get("CONVMINDS_HOME", "."), "models", "huth_alignment")
    os.makedirs(save_dir, exist_ok=True)
    mode_str = "vae" if args.use_vae else "mlp"
    save_path = os.path.join(save_dir, f"huth_{mode_str}_{args.loss_type}.pt")

    # 4. Check for cached model
    if os.path.exists(save_path):
        logger.info(f"Found cached model at {save_path}. Skipping training.")
        model.load_state_dict(torch.load(save_path, map_location=device))
        skip_training = True
    else:
        skip_training = False

    # 5. Training logic
    if not skip_training:
        phase_epochs = [int(e) for e in args.epochs.split(",")]
        total_epochs = sum(phase_epochs)
        
        for epoch in range(1, total_epochs + 1):
            model.train()
            epoch_losses = []
            # Phase logic: 
            # If '10,10' -> Epoch 1-10 is Phase 1 (VAE Recon if use_vae), 11-20 is Phase 2 (Align)
            if len(phase_epochs) > 1:
                phase = 1 if epoch <= phase_epochs[0] else 2
            else:
                phase = 2 # Default to alignment if single epoch count provided
                
            pbar = tqdm(train_loader, desc=f"Ep {epoch:2d} | Ph {phase}")
            for batch in pbar:
                x, ctx, target = batch["bold"].to(device), batch["context"], batch["target"]
                h_text = embedder.embed_with_context(ctx, target).to(device)
                
                outputs = model(x)
                loss_dict = {}
                
                if args.use_vae and phase == 1:
                    # Reconstruction only
                    res = vae_reconstruction_loss(outputs["x_hat"], outputs["x_orig"], outputs["mu"], outputs["logvar"])
                    loss = res["loss"]
                    loss_dict = {"loss": loss.item(), "rec": res["rec_loss"].item()}
                else:
                    # Alignment (+ Recon choice)
                    if args.loss_type == "info_nce":
                        align_loss = info_nce_loss(outputs["h_llm"], h_text)
                    else:
                        align_loss = F.mse_loss(outputs["h_llm"], h_text)
                    
                    loss = align_loss
                    if args.use_vae:
                        res = vae_reconstruction_loss(outputs["x_hat"], outputs["x_orig"], outputs["mu"], outputs["logvar"])
                        loss = loss + res["loss"]
                        loss_dict["rec"] = res["rec_loss"].item()
                    
                    loss_dict["align"] = align_loss.item()
                    loss_dict["loss"] = loss.item()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_losses.append(loss.item())
                pbar.set_postfix(loss_dict)

            # End of epoch summary
            avg_loss = np.mean(epoch_losses)
            logger.info(f"Epoch {epoch:2d}/{total_epochs} completed | Phase {phase} | Avg Loss: {avg_loss:.4f}")

        # Final Save
        torch.save(model.state_dict(), save_path)
        logger.info(f"\nModel saved to: {save_path}")

    # 6. Final Evaluation
    logger.info("\n" + "="*40)
    logger.info("FINAL PERFORMANCE REPORT")
    logger.info("="*40)
    stats = run_eval(model, test_loader, embedder, device)
    for k, v in stats.items():
        fmt = f"{v*100:.2f}%" if k in ["top10_acc", "bleu", "rouge1", "rougeL", "meteor"] else f"{v:.4f}"
        logger.info(f"{k.upper():<12}: {fmt}")
    
    # 6. Visual Samples
    logger.info("\n" + "="*40)
    logger.info("DECODING SAMPLES")
    logger.info("="*40)
    model.eval()
    with torch.no_grad():
        # Just grab one batch for samples
        batch = next(iter(test_loader))
        x, ctx, target = batch["bold"].to(device), batch["context"], batch["target"]
        outputs = model(x)
        h_pred = outputs["h_llm"]
        
        # Retrieval for sample 1
        scores = F.cosine_similarity(h_pred[0:1], h_pred, dim=-1) # self-check just for visual
        top_tokens = embedder.predict_tokens(h_pred[:3], top_k=3)
        
        for i in range(3):
            logger.info(f"\n[{i+1}] {batch['subject'][i]} | {batch['story'][i]} | TR {batch['tr'][i]}")
            logger.info(f"  CONTEXT  : {ctx[i][:100]}...")
            logger.info(f"  TRUE     : {target[i]}")
            # top brain tokens
            tok_str = ", ".join([f"{t} ({p*100:.1f}%)" for t, p in top_tokens[i]])
            logger.info(f"  PREDICTED: {tok_str}")

if __name__ == "__main__":
    main()
