import torch
import torch.nn as nn
import torch.nn.functional as F

class BrainSteerAdapter(nn.Module):
    """
    Adapter that maps brain activity (4 TRs) to a steering vector 
    via cross-attention with the LLM's current hidden state.
    
    Architecture:
    - Trainable Positional Embeddings for the 4 TRs
    - Multi-head Cross-Attention (LLM as Query, Brain as Key/Value)
    - MLP Output Head (1 Layer + GELU)
    
    Total Parameters: ~7.86M (for d_model=768)
    """
    def __init__(self, brain_dim=1000, llm_dim=768, num_heads=12, n_frames=4):
        super().__init__()
        self.brain_dim = brain_dim
        self.llm_dim = llm_dim
        self.n_frames = n_frames
        
        # 1. Positional Embedding
        self.pos_embed = nn.Parameter(torch.randn(1, n_frames, brain_dim) * 0.02)
        
        # 2. Projections
        self.W_K = nn.Linear(brain_dim, llm_dim)
        self.W_V = nn.Linear(brain_dim, llm_dim)
        self.W_Q = nn.Linear(llm_dim, llm_dim)
        
        # 3. Cross-Attention
        self.attn = nn.MultiheadAttention(
            embed_dim=llm_dim, 
            num_heads=num_heads, 
            batch_first=True
        )
        
        # 4. Transformer MLP (Output Head)
        self.mlp = nn.Sequential(
            nn.Linear(llm_dim, llm_dim * 4), # 768 -> 3072
            nn.GELU(),
            nn.Linear(llm_dim * 4, llm_dim)  # 3072 -> 768
        )

    def forward(self, B, H_query):
        """
        Args:
            B: Brain Input [Batch, 4, 1000]
            H_query: LLM Hidden State [Batch, 1, 768] (last token of prompt)
            
        Returns:
            v_steer: The steering vector to be added to the residual stream [Batch, 1, 768]
        """
        # 1. Add positional information to brain TRs
        B = B + self.pos_embed
        
        # 2. Project to attention space
        K = self.W_K(B) # [Batch, 4, 768]
        V = self.W_V(B) # [Batch, 4, 768]
        Q = self.W_Q(H_query) # [Batch, 1, 768]
        
        # 3. Cross-Attention
        # Query: LLM state | Key/Value: Brain states
        A, weights = self.attn(query=Q, key=K, value=V)
        
        # 4. Post-attention MLP
        v_steer = self.mlp(A)
        
        return v_steer

if __name__ == "__main__":
    # Quick shape check
    batch_size = 8
    brain_dim = 1000
    llm_dim = 768
    n_frames = 4
    
    adapter = BrainSteerAdapter(brain_dim, llm_dim, n_frames=n_frames)
    
    B = torch.randn(batch_size, n_frames, brain_dim)
    H_query = torch.randn(batch_size, 1, llm_dim)
    
    v_steer = adapter(B, H_query)
    print(f"Output shape: {v_steer.shape}") # Expected: [8, 1, 768]
    
    total_params = sum(p.numel() for p in adapter.parameters())
    print(f"Total Parameters: {total_params / 1e6:.2f}M")
