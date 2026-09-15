#!/usr/bin/env python3
import sys
import tiktoken
import torch
import torch.nn.functional as F

from train import GPT, GPTConfig


def generate_completion(
    model,
    enc,
    prompt: str,
    max_new_tokens: int = 50,
    temperature: float = 0.7,
    top_k: int = 40,
    device: str = "cpu",
) -> str:
    tokens = enc.encode(prompt)
    x = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)

    model.eval()
    with torch.no_grad():
        for _ in range(max_new_tokens):
            # Crop to context window if length exceeds 256
            x_cond = x if x.size(1) <= 256 else x[:, -256:]

            logits, _ = model(x_cond)
            logits = logits[:, -1, :] / temperature

            # Apply top-k filtering
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("Inf")

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            x = torch.cat((x, next_token), dim=1)

    return enc.decode(x[0].tolist())


def main():
    ckpt_path = sys.argv[1] if len(sys.argv) > 1 else "checkpoint_best.pt"
    device = "cpu"

    print(f"Loading weights from {ckpt_path}...")
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)

    cfg = GPTConfig(
        vocab_size=50257,
        n_layer=6,
        n_head=6,
        n_embd=384,
        block_size=256,
    )
    model = GPT(cfg)

    state_dict = (
        checkpoint["model_state_dict"]
        if "model_state_dict" in checkpoint
        else checkpoint
    )
    model.load_state_dict(state_dict)
    model.to(device)

    enc = tiktoken.get_encoding("gpt2")

    test_prompts = [
        "It was a dark and stormy",
        "Sherlock Holmes sat quietly in his chair, smoking his",
        "def parse_arguments(",
        "import os\nimport sys\n\ndef main():\n   ",
    ]

    for i, p in enumerate(test_prompts, start=1):
        print(f"\n{'=' * 20} Test Prompt {i} {'=' * 20}")
        output = generate_completion(
            model,
            enc,
            prompt=p,
            max_new_tokens=50,
            temperature=0.7,
            top_k=40,
            device=device,
        )
        print(output)
        print("=" * 55)


if __name__ == "__main__":
    main()
