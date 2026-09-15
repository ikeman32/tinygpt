#!/usr/bin/env python3
import tiktoken, torch
from model import GPT, GPTConfig

enc = tiktoken.get_encoding("gpt2")
ckpt = torch.load("checkpoint_best.pt", map_location="cpu", weights_only=False)
model = GPT(ckpt.get("config", GPTConfig()))
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

# Raw completion test (no tags)
tokens = enc.encode('def add(a, b):\n    """Return the sum of a and b."""\n    return ')
x = torch.tensor([tokens], dtype=torch.long)

with torch.no_grad():
    for _ in range(40):
        logits, _ = model(x[:, -256:])
        logits = logits[:, -1, :]

        # Light repetition penalty to prevent *args loops
        for token_id in set(x[0].tolist()):
            if logits[0, token_id] > 0:
                logits[0, token_id] /= 1.2
            else:
                logits[0, token_id] *= 1.2

        # Top-k sampling
        v, _ = torch.topk(logits, min(30, logits.size(-1)))
        logits[logits < v[:, [-1]]] = -float("Inf")
        probs = torch.softmax(logits / 0.7, dim=-1)
        next_tok = torch.multinomial(probs, num_samples=1)

        x = torch.cat((x, next_tok), dim=1)

print(enc.decode(x[0].tolist()))
