import torch

def make_pad_mask(seq, pad_idx):
    # Expand to [Batch, 1, 1, Seq_len] to broadcast across heads
    return (seq != pad_idx).unsqueeze(1).unsqueeze(2)

def make_causal_mask(seq_len, device):
    # Lower triangular matrix: [Seq_len, Seq_len]
    return torch.tril(torch.ones(seq_len, seq_len)).type(torch.bool).to(device)