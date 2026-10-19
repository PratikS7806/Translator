import torch
import torch.nn as nn
from mytransformer.model import Transformer
from mytransformer.masks import make_pad_mask, make_causal_mask
from mytransformer.data import get_dataloader

def main():
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load data and tokenizer
    train_loader, sp = get_dataloader(batch_size=64)
    VOCAB_SIZE = sp.get_piece_size()
    PAD_IDX = sp.pad_id()

    # Initialize "Small" Model
    model = Transformer(
        src_vocab=VOCAB_SIZE, tgt_vocab=VOCAB_SIZE, 
        d_model=256, heads=4, num_layers=3, d_ff=1024, dropout=0.1
    ).to(device)

    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005, betas=(0.9, 0.98), eps=1e-9)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX, label_smoothing=0.1)

    EPOCHS = 30
    print(f"Starting training for {EPOCHS} epochs...")
    model.train()

    for epoch in range(EPOCHS):
        total_loss = 0
        for batch_idx, (src_data, tgt_data) in enumerate(train_loader):
            src_data, tgt_data = src_data.to(device), tgt_data.to(device)
            
            tgt_input = tgt_data[:, :-1]
            tgt_expected = tgt_data[:, 1:]
            
            src_mask = make_pad_mask(src_data, PAD_IDX).to(device)
            causal_mask = make_causal_mask(tgt_input.size(1), device)
            tgt_pad_mask = make_pad_mask(tgt_input, PAD_IDX).to(device)
            tgt_mask = causal_mask & tgt_pad_mask
            
            optimizer.zero_grad()
            logits = model(src_data, tgt_input, src_mask, tgt_mask)
            loss = criterion(logits.reshape(-1, VOCAB_SIZE), tgt_expected.reshape(-1))
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item()
            if batch_idx % 100 == 0:
                print(f"Epoch {epoch+1}/{EPOCHS} | Batch {batch_idx}/{len(train_loader)} | Loss: {loss.item():.4f}")
                
        avg_loss = total_loss / len(train_loader)
        print(f"--- Epoch {epoch+1} Completed | Average Loss: {avg_loss:.4f} ---")
        
        # Save checkpoint
        torch.save(model.state_dict(), "translation_model.pt")

    print("Training complete. Model saved as 'translation_model.pt'.")

if __name__ == "__main__":
    main()