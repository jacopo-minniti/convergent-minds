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

# convminds imports
from convminds.data.benchmarks.huth_alignment import HuthAlignmentDataset
from convminds.models.residual_steer import ResidualSteerLM
from convminds.nn.metrics import calculate_nlp_metrics, identification_accuracy

# Configure logging
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(name)s: %(message)s', handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Convergent Minds: Phase-based Brain Steering")
    parser.add_argument("--epochs", type=str, default="5,10", help="Phase 1 and Phase 2 epochs (e.g., '5,10')")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate for the adapter")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for training")
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    phase_epochs = [int(e) for e in args.epochs.split(",")]
    
    # 1. Load Data
    logger.info("Initializing Huth Alignment Dataset...")
    train_set = HuthAlignmentDataset(subject_ids=["S1"], split="train")
    test_set = HuthAlignmentDataset(subject_ids=["S1"], split="test")
    
    # We need a small batch size for the complex forward passes if memory is limited, 
    # but 32 should be fine for GPT-2 Small.
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)
    
    # 2. Setup Model
    logger.info("Initializing ResidualSteerLM (GPT-2 Small)...")
    model = ResidualSteerLM(llm_id="gpt2", injection_layer=6).to(device)
    optimizer = AdamW(model.adapter.parameters(), lr=args.lr) # Only train the adapter
    
    # 3. Phase 1: The Warmup (MSE Alignment)
    if phase_epochs[0] > 0:
        logger.info(f"Starting Phase 1: MSE Warmup ({phase_epochs[0]} epochs)")
        for epoch in range(1, phase_epochs[0] + 1):
            model.adapter.train()
            epoch_losses = []
            pbar = tqdm(train_loader, desc=f"Ph1 Ep {epoch}")
            for batch in pbar:
                B = batch["bold"].to(device) # [Batch, 4, 1000]
                contexts = batch["context"]
                targets = batch["target"]
                
                # Tokenize
                if isinstance(contexts, str): 
                    logger.warning(f"Forcing string context to list: '{contexts[:20]}...'")
                    contexts = [contexts]
                if isinstance(targets, str): 
                    logger.warning(f"Forcing string target to list: '{targets[:20]}...'")
                    targets = [targets]
                ctx_ids = model.tokenizer(contexts, return_tensors="pt", padding=True, truncation=True).input_ids.to(device)
                tgt_ids = model.tokenizer(targets, return_tensors="pt", padding=True, truncation=True).input_ids.to(device)
                
                with torch.no_grad():
                    # Query Extraction (last token of context)
                    H_query_full = model.get_h_at_layer(ctx_ids)
                    H_query = H_query_full[:, -1:, :] # [Batch, 1, 768]
                    
                    # Target Extraction (average of target words)
                    # NOTE: We pass target words alone (perceived continuation) as per spec
                    H_target_full = model.get_h_at_layer(tgt_ids)
                    # Simple average over sequence length for the target words
                    H_target = H_target_full.mean(dim=1, keepdim=True) # [Batch, 1, 768]
                    
                    delta_target = H_target - H_query
                
                # Forward Adapter
                # We want v_steer to represent the delta
                v_steer = model.adapter(B, H_query)
                
                loss = F.mse_loss(v_steer, delta_target)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                epoch_losses.append(loss.item())
                pbar.set_postfix({"mse": f"{loss.item():.4f}"})
            
            logger.info(f"Phase 1 Epoch {epoch} Avg Loss: {np.mean(epoch_losses):.4f}")

    # 4. Phase 2: Main Training (Cross-Entropy & Injection)
    if len(phase_epochs) > 1 and phase_epochs[1] > 0:
        logger.info(f"Starting Phase 2: CE Injection ({phase_epochs[1]} epochs)")
        # CE Loss
        criterion = torch.nn.CrossEntropyLoss()
        
        for epoch in range(1, phase_epochs[1] + 1):
            model.adapter.train()
            epoch_losses = []
            pbar = tqdm(train_loader, desc=f"Ph2 Ep {epoch}")
            for batch in pbar:
                B = batch["bold"].to(device)
                contexts = batch["context"]
                targets = batch["target"]
                
                # We need to predict the FIRST token of the target given the context
                # So we concatenate context + target[0] for the forward pass, 
                # but the loss is only on the prediction at the end of the context.
                
                # Actually, the spec says: 
                # "Pass your text prompt token IDs... stop at layer 6... inject... resume... loss against actual target word ID"
                
                # Let's tokenize context and get the ID of the first target token
                # This ensures we are predicting the next word correctly.
                if isinstance(contexts, str): 
                    logger.warning(f"Forcing string context to list in Ph2: '{contexts[:20]}...'")
                    contexts = [contexts]
                input_ids = model.tokenizer(contexts, return_tensors="pt", padding=True, truncation=True).input_ids.to(device)
                
                # Get the true next token ID
                # We take the first token of each target string
                target_tokens = [model.tokenizer.encode(t)[0] if len(t) > 0 else model.tokenizer.eos_token_id for t in targets]
                target_label = torch.tensor(target_tokens, device=device)
                
                # Forward Steered pass
                logits, v_steer = model.forward_steered(input_ids, B)
                
                # Grab the logits for the last token of the context
                next_token_logits = logits[:, -1, :] # [Batch, Vocab]
                
                loss = criterion(next_token_logits, target_label)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                epoch_losses.append(loss.item())
                pbar.set_postfix({"ce_loss": f"{loss.item():.4f}"})
            
            logger.info(f"Phase 2 Epoch {epoch} Avg Loss: {np.mean(epoch_losses):.4f}")

    # 5. Final Evaluation (Simple check)
    logger.info("Evaluation on test set...")
    model.adapter.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for batch in test_loader:
            B = batch["bold"].to(device)
            contexts = batch["context"]
            targets = batch["target"]
            
            if isinstance(contexts, str): 
                logger.warning(f"Forcing string context to list in Eval: '{contexts[:20]}...'")
                contexts = [contexts]
            if isinstance(targets, str): 
                logger.warning(f"Forcing string target to list in Eval: '{targets[:20]}...'")
                targets = [targets]
            
            input_ids = model.tokenizer(contexts, return_tensors="pt", padding=True, truncation=True).input_ids.to(device)
            target_tokens = [model.tokenizer.encode(t)[0] if len(t) > 0 else model.tokenizer.eos_token_id for t in targets]
            target_label = torch.tensor(target_tokens, device=device)
            
            logits, _ = model.forward_steered(input_ids, B)
            next_token_logits = logits[:, -1, :]
            preds = torch.argmax(next_token_logits, dim=-1)
            
            correct += (preds == target_label).sum().item()
            total += target_label.size(0)
            
        logger.info(f"Zero-shot Top-1 Accuracy: {100 * correct / total:.2f}%")

    # Save model
    save_path = "brain_steer_gpt2.pt"
    torch.save(model.adapter.state_dict(), save_path)
    logger.info(f"Adapter saved to {save_path}")

if __name__ == "__main__":
    main()
