import os
import sys
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
import numpy as np
from tqdm import tqdm
import logging
import argparse

from convminds.data.benchmarks.huth_alignment import HuthAlignmentDataset
from convminds.models.residual_steer import ResidualSteerLM
from convminds.nn.metrics import calculate_nlp_metrics, identification_accuracy

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(name)s: %(message)s', handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Convergent Minds: Phase-based Brain Steering")
    parser.add_argument("--epochs", type=str, default="5,10")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    phase_epochs = [int(e) for e in args.epochs.split(",")]
    
    logger.info("Initializing Huth Alignment Dataset...")
    train_set = HuthAlignmentDataset(subject_ids=["S1"], split="train")
    test_set = HuthAlignmentDataset(subject_ids=["S1"], split="test")
    
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)
    
    logger.info("Initializing ResidualSteerLM (GPT-2 Small)...")
    model = ResidualSteerLM(llm_id="gpt2", injection_layer=6).to(device)
    model.tokenizer.padding_side = "left" # Ensure [:, -1, :] points to the last context token
    optimizer = AdamW(model.adapter.parameters(), lr=args.lr)
    
    # --- PHASE 1 ---
    if phase_epochs[0] > 0:
        logger.info(f"Starting Phase 1: MSE Warmup ({phase_epochs[0]} epochs)")
        for epoch in range(1, phase_epochs[0] + 1):
            model.adapter.train()
            epoch_losses = []
            pbar = tqdm(train_loader, desc=f"Ph1 Ep {epoch}")
            for batch in pbar:
                B = batch["bold"].to(device)
                
                # Tokenizer natively handles tuples of strings from DataLoader
                ctx_enc = model.tokenizer(batch["context"], return_tensors="pt", padding=True, truncation=True)
                tgt_enc = model.tokenizer(batch["target"], return_tensors="pt", padding=True, truncation=True)
                
                ctx_ids, ctx_mask = ctx_enc.input_ids.to(device), ctx_enc.attention_mask.to(device)
                tgt_ids, tgt_mask = tgt_enc.input_ids.to(device), tgt_enc.attention_mask.to(device)
                
                with torch.no_grad():
                    H_query = model.get_h_at_layer(ctx_ids, attention_mask=ctx_mask)[:, -1:, :]
                    
                    # Masked mean for target signal
                    H_target_raw = model.get_h_at_layer(tgt_ids, attention_mask=tgt_mask)
                    mask_expanded = tgt_mask.unsqueeze(-1).float()
                    sum_h = torch.sum(H_target_raw * mask_expanded, dim=1, keepdim=True)
                    count = torch.sum(mask_expanded, dim=1, keepdim=True).clamp(min=1e-8)
                    H_target = sum_h / count
                    
                    delta_target = H_target - H_query
                
                v_steer = model.adapter(B, H_query)
                loss = F.mse_loss(v_steer, delta_target)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                epoch_losses.append(loss.item())
                pbar.set_postfix({"mse": f"{loss.item():.4f}"})

            avg_loss = np.mean(epoch_losses)
            logger.info(f"Phase 1 | Epoch {epoch} Completed | Average MSE Loss: {avg_loss:.6f}")

    # --- PHASE 2 ---
    if len(phase_epochs) > 1 and phase_epochs[1] > 0:
        logger.info(f"Starting Phase 2: CE Injection ({phase_epochs[1]} epochs)")
        criterion = torch.nn.CrossEntropyLoss()
        
        for epoch in range(1, phase_epochs[1] + 1):
            model.adapter.train()
            epoch_losses = []
            pbar = tqdm(train_loader, desc=f"Ph2 Ep {epoch}")
            for batch in pbar:
                B = batch["bold"].to(device)
                ctx_enc = model.tokenizer(batch["context"], return_tensors="pt", padding=True, truncation=True)
                input_ids, attention_mask = ctx_enc.input_ids.to(device), ctx_enc.attention_mask.to(device)
                
                # Encode target with space prefix to align with expected next-token distribution
                target_tokens = [model.tokenizer.encode(" " + t)[0] if len(t) > 0 else model.tokenizer.eos_token_id for t in batch["target"]]
                target_label = torch.tensor(target_tokens, device=device)
                
                logits, _ = model.forward_steered(input_ids, B, attention_mask=attention_mask)
                next_token_logits = logits[:, -1, :]
                
                loss = criterion(next_token_logits, target_label)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                epoch_losses.append(loss.item())
                pbar.set_postfix({"ce_loss": f"{loss.item():.4f}"})

            avg_loss = np.mean(epoch_losses)
            logger.info(f"Phase 2 | Epoch {epoch} Completed | Average CE Loss: {avg_loss:.6f}")

    # --- EVALUATION ---
    logger.info("Evaluation on test set...")
    model.adapter.eval()
    with torch.no_grad():
        correct = total = 0
        for batch in tqdm(test_loader, desc="Evaluation"):
            B = batch["bold"].to(device)
            ctx_enc = model.tokenizer(batch["context"], return_tensors="pt", padding=True, truncation=True)
            input_ids, attention_mask = ctx_enc.input_ids.to(device), ctx_enc.attention_mask.to(device)
            
            target_tokens = [model.tokenizer.encode(" " + t)[0] if len(t) > 0 else model.tokenizer.eos_token_id for t in batch["target"]]
            target_label = torch.tensor(target_tokens, device=device)
            
            logits, _ = model.forward_steered(input_ids, B, attention_mask=attention_mask)
            preds = torch.argmax(logits[:, -1, :], dim=-1)
            
            correct += (preds == target_label).sum().item()
            total += target_label.size(0)
            
        logger.info(f"Zero-shot Top-1 Accuracy: {100 * correct / total:.2f}%")

    save_path = "brain_steer_gpt2.pt"
    torch.save(model.adapter.state_dict(), save_path)
    logger.info(f"Adapter saved to {save_path}")

if __name__ == "__main__":
    main()