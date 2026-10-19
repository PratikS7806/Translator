import torch
import torch.nn as nn
import math
from .attention import MultiHeadAttention
from .layers import FeedForward, PositionalEncoding

class EncoderLayer(nn.Module):
    def __init__(self, d_model, heads, d_ff, dropout=0.1):
        super().__init__()
        self.attn = MultiHeadAttention(d_model, heads)
        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask):
        x = x + self.dropout(self.attn(self.norm1(x), self.norm1(x), self.norm1(x), mask))
        x = x + self.dropout(self.ffn(self.norm2(x)))
        return x

class DecoderLayer(nn.Module):
    def __init__(self, d_model, heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, heads)
        self.cross_attn = MultiHeadAttention(d_model, heads)
        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, memory, src_mask, tgt_mask):
        x = x + self.dropout(self.self_attn(self.norm1(x), self.norm1(x), self.norm1(x), tgt_mask))
        x = x + self.dropout(self.cross_attn(self.norm2(x), memory, memory, src_mask))
        x = x + self.dropout(self.ffn(self.norm3(x)))
        return x

class Transformer(nn.Module):
    def __init__(self, src_vocab, tgt_vocab, d_model=64, heads=4, num_layers=2, d_ff=256, dropout=0.1):
        super().__init__()
        self.src_embed = nn.Embedding(src_vocab, d_model)
        self.tgt_embed = nn.Embedding(tgt_vocab, d_model)
        self.pos_enc = PositionalEncoding(d_model)
        
        self.encoder = nn.ModuleList([EncoderLayer(d_model, heads, d_ff, dropout) for _ in range(num_layers)])
        self.decoder = nn.ModuleList([DecoderLayer(d_model, heads, d_ff, dropout) for _ in range(num_layers)])
        
        self.final_proj = nn.Linear(d_model, tgt_vocab)
        self.final_proj.weight = self.tgt_embed.weight
        self.dropout = nn.Dropout(dropout)
        
    def encode(self, src, src_mask):
        x = self.dropout(self.pos_enc(self.src_embed(src) * math.sqrt(self.src_embed.embedding_dim)))
        for layer in self.encoder:
            x = layer(x, src_mask)
        return x
        
    def decode(self, tgt, memory, src_mask, tgt_mask):
        x = self.dropout(self.pos_enc(self.tgt_embed(tgt) * math.sqrt(self.tgt_embed.embedding_dim)))
        for layer in self.decoder:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.final_proj(x)
        
    def forward(self, src, tgt, src_mask, tgt_mask):
        memory = self.encode(src, src_mask)
        return self.decode(tgt, memory, src_mask, tgt_mask)