import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel

class GPT2Embedder:
    """
    Utility for extracting context-aware semantic embeddings from GPT-2.
    It can also decode latent vectors back into top-k tokens via the LM head.
    """
    def __init__(self, device="cpu"):
        self.device = device
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        self.model = GPT2LMHeadModel.from_pretrained("gpt2").to(self.device)
        self.model.eval()

    @torch.no_grad()
    def embed_with_context(self, context_texts, target_texts):
        """
        Embeds the target_texts given the context_texts. 
        Returns the mean of the hidden states for the tokens in target_text.
        """
        all_embeddings = []
        for ctx, target in zip(context_texts, target_texts):
            full_text = (ctx.strip() + " " + target.strip()).strip()
            if not full_text:
                all_embeddings.append(torch.zeros(1, 768).to(self.device))
                continue
                
            inputs = self.tokenizer(full_text, return_tensors="pt").to(self.device)
            # transformer is the core GPT2 model
            outputs = self.model.transformer(**inputs)
            hidden_states = outputs.last_hidden_state # (Batch, SeqLen, 768)
            
            target_ids = self.tokenizer(target.strip(), add_special_tokens=False).input_ids
            num_target_tokens = len(target_ids)
            
            if num_target_tokens > 0:
                # Take the last N tokens (the target)
                target_hidden = hidden_states[:, -num_target_tokens:, :]
                mean_target = target_hidden.mean(dim=1)
            else:
                # If target is empty, use the last state of the context
                mean_target = hidden_states[:, -1:, :]
                
            all_embeddings.append(mean_target.squeeze(1))
            
        return torch.cat(all_embeddings, dim=0)

    @torch.no_grad()
    def predict_tokens(self, latent_vecs, top_k=5):
        """
        Passes the latent vectors through the GPT-2 LM head to get token probabilities.
        """
        logits = self.model.lm_head(latent_vecs)
        probs = torch.softmax(logits, dim=-1)
        
        top_probs, top_indices = torch.topk(probs, k=top_k, dim=-1)
        
        results = []
        for b in range(latent_vecs.shape[0]):
            toks = [self.tokenizer.decode([idx]).strip() for idx in top_indices[b]]
            scores = top_probs[b].tolist()
            results.append(list(zip(toks, scores)))
        return results
