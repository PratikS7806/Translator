import torch
import torch.nn as nn
import math

def scaled_dot_product_attention(q, k, v, mask=None):
    d_k = q.size(-1)
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))
    attn = torch.softmax(scores, dim=-1)
    return torch.matmul(attn, v), attn

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, heads):
        super().__init__()
        self.heads = heads
        self.d_k = d_model // heads
        
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
    def forward(self, q, k, v, mask=None):
        B = q.size(0)
        q = self.q_proj(q).view(B, -1, self.heads, self.d_k).transpose(1, 2)
        k = self.k_proj(k).view(B, -1, self.heads, self.d_k).transpose(1, 2)
        v = self.v_proj(v).view(B, -1, self.heads, self.d_k).transpose(1, 2)
        
        context, _ = scaled_dot_product_attention(q, k, v, mask)
        context = context.transpose(1, 2).contiguous().view(B, -1, self.heads * self.d_k)
        return self.out_proj(context)