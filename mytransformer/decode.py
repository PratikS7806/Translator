import torch
import torch.nn.functional as F
from .masks import make_causal_mask

def beam_search_decode(model, src, src_mask, max_len, bos_idx, eos_idx, device, beam_width=3):
    model.eval()
    with torch.no_grad():
        # Process the source text once to get the encoder memory
        memory = model.encode(src, src_mask)
        
        # Initialize the beam with the BOS token and a starting score of 0.0
        # Format: list of tuples (sequence_tensor, cumulative_log_score)
        beam = [(torch.ones(1, 1).fill_(bos_idx).type_as(src.data), 0.0)]
        
        for _ in range(max_len - 1):
            new_beam = []
            
            for seq, score in beam:
                # If a sequence already hit the EOS token, pass it through unchanged
                if seq[0, -1].item() == eos_idx:
                    new_beam.append((seq, score))
                    continue
                    
                tgt_mask = make_causal_mask(seq.size(1), device)
                out = model.decode(seq, memory, src_mask, tgt_mask)
                
                # Extract the last predicted position and convert to log probabilities
                log_probs = F.log_softmax(out[:, -1], dim=-1)
                
                # Get the top 'beam_width' candidates for this specific sequence
                top_scores, top_indices = torch.topk(log_probs, beam_width, dim=1)
                
                for i in range(beam_width):
                    next_word = top_indices[0, i].item()
                    next_score = top_scores[0, i].item()
                    
                    # Append the new word to the sequence and add the log probabilities
                    new_seq = torch.cat([seq, torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=1)
                    new_beam.append((new_seq, score + next_score))
            
            # Sort all generated branches by their cumulative score descending
            beam = sorted(new_beam, key=lambda x: x[1], reverse=True)[:beam_width]
            
            # Stop early if the top 'beam_width' sequences have all reached the EOS token
            if all(seq[0, -1].item() == eos_idx for seq, _ in beam):
                break
                
        # Return only the sequence tensor of the highest-scoring complete sentence
        best_seq, best_score = beam[0]
        return best_seq