import torch
import torch.nn as nn
from mytransformer.model import Transformer
from mytransformer.masks import make_pad_mask, make_causal_mask
from mytransformer.decode import greedy_decode

def main():
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    PAD_IDX = 0
    VOCAB_SIZE = 20
    SEQ_LEN = 10
    BATCH_SIZE = 32
    
    model = Transformer(VOCAB_SIZE, VOCAB_SIZE).to(device)
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
            
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.98), eps=1e-9)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX, label_smoothing=0.1) 
    
    src_data = torch.randint(1, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN)).to(device)
    tgt_data = src_data.clone() 
    tgt_input = tgt_data[:, :-1]
    tgt_expected = tgt_data[:, 1:]
    
    src_mask = make_pad_mask(src_data, PAD_IDX).to(device)
    causal_mask = make_causal_mask(tgt_input.size(1), device)
    tgt_pad_mask = make_pad_mask(tgt_input, PAD_IDX).to(device)
    tgt_mask = causal_mask & tgt_pad_mask
    
    model.train()
    for epoch in range(600):
        optimizer.zero_grad()
        logits = model(src_data, tgt_input, src_mask, tgt_mask)
        loss = criterion(logits.reshape(-1, VOCAB_SIZE), tgt_expected.reshape(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        if epoch % 100 == 0:
            print(f"Epoch {epoch:03d} | Loss: {loss.item():.4f}")
            
    print("Testing Decode:")
    test_src = src_data[0].unsqueeze(0) 
    test_src_mask = make_pad_mask(test_src, PAD_IDX).to(device)
    output_seq = greedy_decode(model, test_src, test_src_mask, 15, 1, 2, device)
    print("Expected:", tgt_data[0].cpu().numpy())
    print("Generated:", output_seq[0].cpu().numpy())

if __name__ == "__main__":
    main()