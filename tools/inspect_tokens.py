#!/usr/bin/env python3
import sys
import tiktoken
import torch
import torch.nn.functional as F

# Import your model definition from your training file
from train import GPT, GPTConfig


def inspect_next_tokens(
    model, enc, prompt_text: str, top_k: int = 5, device: str = "cpu"
):
    tokens = enc.encode(prompt_text)
    x = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)

    with torch.no_grad():
        logits, _ = model(x)
        # Focus on the logits for the final token position
        next_token_logits = logits[0, -1, :]
        probs = F.softmax(next_token_logits, dim=-1)
        top_probs, top_indices = torch.topk(probs, top_k)

    print(f"\nPrompt: {repr(prompt_text)}")
    print(f"{'-' * 45}")
    for rank, (p, idx) in enumerate(
        zip(top_probs.tolist(), top_indices.tolist()), start=1
    ):
        token_str = enc.decode([idx])
        print(f" {rank}. {repr(token_str):<15} | Prob: {p:.4f} ({p * 100:.1f}%)")


def main():
    ckpt_path = sys.argv[1] if len(sys.argv) > 1 else "checkpoint_best.pt"
    device = "cpu"

    print(f"Loading weights from {ckpt_path}...")
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)

    # 6 layers, 6 heads, 384 embedding dim, 256 block size
    cfg = GPTConfig(
        vocab_size=50257,
        n_layer=6,
        n_head=6,
        n_embd=384,
        block_size=256,
    )
    model = GPT(cfg)

    # Support raw state dict or wrapped checkpoint dict
    state_dict = (
        checkpoint["model_state_dict"]
        if "model_state_dict" in checkpoint
        else checkpoint
    )
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    enc = tiktoken.get_encoding("gpt2")

    # Test Prompts across both corpus domains
    test_prompts = [
        # Literature domain
        "It was a dark and stormy",
        'Sherlock Holmes sat quietly in his chair, smoking his',
        # Technical / Code domain
        "def parse_arguments(",
        "import os\nimport sys\n\ndef main():\n   ",
    ]

    for p in test_prompts:
        inspect_next_tokens(model, enc, p, top_k=5, device=device)


if __name__ == "__main__":
    main()
