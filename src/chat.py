#!/usr/bin/env python3
"""
chat.py - Interactive CLI interface for the fine-tuned 30M model.
"""

from typing import cast
import torch
import torch.nn as nn
import tiktoken
from model import GPT, GPTConfig

CHECKPOINT_PATH = "./ckpt-fine-tuned/checkpoint_sft.pt"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CONTEXT_LENGTH = 256

USER_TAG = "User: "
ASSISTANT_TAG = "\nAssistant:\n"  # Clean boundary: no trailing whitespace token 220


def generate_response(
    model: GPT,
    enc: tiktoken.Encoding,
    prompt: str,
    max_new_tokens: int = 150,
    temperature: float = 0.2,
    top_k: int = 15,
    repetition_penalty: float = 1.15,
    confidence_threshold: float = 0.25,  # Minimum confidence required on token 0
) -> str:
    model.eval()
    clean_prompt = " ".join(prompt.strip().split())
    formatted = f"{USER_TAG}{clean_prompt}{ASSISTANT_TAG}"
    tokens = enc.encode(formatted, allowed_special={"<|endoftext|>"})
    idx = torch.tensor([tokens], dtype=torch.long, device=DEVICE)

    generated = []

    with torch.no_grad():
        for step in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= CONTEXT_LENGTH else idx[:, -CONTEXT_LENGTH:]
            logits, _ = model(idx_cond)
            logits = logits[:, -1, :]

            # --- Check confidence on the very first token (step 0) ---
            if step == 0:
                raw_probs = torch.softmax(logits, dim=-1)
                top_prob, _ = torch.max(raw_probs, dim=-1)
                if top_prob.item() < confidence_threshold:
                    return "I don't know the answer to that."
            # ---------------------------------------------------------

            # Repetition penalty
            if repetition_penalty != 1.0 and len(generated) > 0:
                for prev_tok in set(generated):
                    if logits[0, prev_tok] > 0:
                        logits[0, prev_tok] /= repetition_penalty
                    else:
                        logits[0, prev_tok] *= repetition_penalty

            # Temperature & top-k
            logits = logits / max(temperature, 1e-5)
            if top_k is not None and top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("Inf")

            probs = torch.softmax(logits, dim=-1)
            next_token = torch.argmax(logits, dim=-1, keepdim=True)

            token_val = next_token.item()
            if token_val == enc.eot_token:
                break

            generated.append(token_val)
            idx = torch.cat((idx, next_token), dim=1)

    return enc.decode(generated).strip()


def main():
    print(f"Loading {CHECKPOINT_PATH} on {DEVICE}...")
    enc = tiktoken.get_encoding("gpt2")
    
    ckpt = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
    config = ckpt.get("config", GPTConfig())
    # print(config)
    model = GPT(config)

    # Handle checkpoint dictionary mapping
    if "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
    elif "model" in ckpt:
        state_dict = ckpt["model"]
    else:
        state_dict = ckpt

    clean_state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(clean_state_dict)

    # Re-tie weights cleanly
    wte_layer = cast(nn.Embedding, model.transformer["wte"])
    model.lm_head.weight = wte_layer.weight

    model.to(DEVICE)
    print("Ready. Type your prompt, or type '/exit' to quit.\n")

    while True:
        try:
            user_input = input("User > ").strip()
            if not user_input:
                continue
            if user_input.lower() in {"/exit", "/quit"}:
                print("Goodbye.")
                break

            response = generate_response(model, enc, user_input)
            print(f"\nAssistant > {response}\n")
        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            break


if __name__ == "__main__":
    main()