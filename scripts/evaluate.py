import torch
import sacrebleu
import sentencepiece as spm
from mytransformer.model import Transformer
from mytransformer.masks import make_pad_mask
from mytransformer.decode import beam_search_decode

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Evaluating on device: {device}")
    
    # 1. Load the SentencePiece Tokenizer
    sp = spm.SentencePieceProcessor(model_file='m30k_bpe.model')
    VOCAB_SIZE = sp.get_piece_size()
    PAD_IDX = sp.pad_id()
    BOS_IDX = sp.bos_id()
    EOS_IDX = sp.eos_id()

    # 2. Load the Trained Model
    model = Transformer(
        src_vocab=VOCAB_SIZE, tgt_vocab=VOCAB_SIZE, 
        d_model=256, heads=4, num_layers=3, d_ff=1024, dropout=0.1
    ).to(device)
    
    # Load the saved weights from training
    model.load_state_dict(torch.load("translation_model.pt", map_location=device))
    model.eval()

    # 3. Load the Validation Dataset
    with open("data/multi30k/val.de", "r", encoding="utf-8") as f:
        sources = [line.strip() for line in f]
    with open("data/multi30k/val.en", "r", encoding="utf-8") as f:
        references = [line.strip() for line in f]

    # Evaluate on the first 500 sentences to save time, or use len(sources) for the full set
    test_size = min(500, len(sources))
    print(f"Starting translation of {test_size} sentences...")
    
    hypotheses = []
    
    for i in range(test_size):
        src_text = sources[i]
        
        # Tokenize source
        src_encoded = [BOS_IDX] + sp.encode_as_ids(src_text) + [EOS_IDX]
        src_tensor = torch.tensor(src_encoded).unsqueeze(0).to(device)
        src_mask = make_pad_mask(src_tensor, PAD_IDX).to(device)
        
        # Decode
        output_ids = beam_search_decode(
        model, 
        src_tensor, 
        src_mask, 
        max_len=50, 
        bos_idx=BOS_IDX, 
        eos_idx=EOS_IDX, 
        device=device,
        beam_width=3
        )
        
        # Clean up special tokens
        out_list = output_ids.squeeze().tolist()
        if isinstance(out_list, int): 
            out_list = [out_list]
        clean_out = [idx for idx in out_list if idx not in (BOS_IDX, EOS_IDX, PAD_IDX)]
        
        # Convert IDs back to English text
        translated_text = sp.decode_ids(clean_out)
        hypotheses.append(translated_text)
        
        # Print 10 examples (good and bad) as required by the plan
        if i < 10:
            print(f"\n--- Example {i+1} ---")
            print(f"Source (DE): {src_text}")
            print(f"Target (EN): {references[i]}")
            print(f"Model  (EN): {translated_text}")
            
    # 4. Compute the sacreBLEU score
    refs_for_bleu = [references[:test_size]]
    bleu = sacrebleu.corpus_bleu(hypotheses, refs_for_bleu)
    
    print("\n" + "="*30)
    print("EVALUATION COMPLETE")
    print(f"sacreBLEU Score: {bleu.score:.2f}")
    print("="*30)

if __name__ == "__main__":
    main()