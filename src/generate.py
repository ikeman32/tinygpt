import torch
import torch.nn.functional as F

def generate(model, idx, max_new_tokens=100, temperature=0.8, top_k=None):
    """
    idx: (B, T) tensor of token indices
    """
    for _ in range(max_new_tokens):
        # Crop context to block_size if it exceeds model context
        idx_cond = idx if idx.size(1) <= model.config.block_size else idx[:, -model.config.block_size:]
        
        logits, _ = model(idx_cond)
        # Pluck logits at the final step and apply temperature
        logits = logits[:, -1, :] / temperature
        
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float("Inf")
            
        probs = F.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1)
        idx = torch.cat((idx, idx_next), dim=1)
        
    return idx