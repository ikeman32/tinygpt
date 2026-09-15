#!/usr/bin/env python3
"""
train_sft.py - Supervised Fine-Tuning (SFT) for 30M Transformer.
Applies prompt loss masking (ignore_index=-100) and causal sequence shifting.
"""

import json
import math
import os
import time
from typing import cast
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import tiktoken

from model import GPT, GPTConfig

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
DATA_PATH = "./sft/sft_data.jsonl"      # Fine tuning training data
CHECKPOINT_IN = "./ckpt-pretrained/checkpoint_best.pt"      # Pretrained checkpoint
CHECKPOINT_OUT = "checkpoint_sft.pt"        # Fine tuned check point to chat with

CONTEXT_LENGTH = 256
BATCH_SIZE = 4
GRAD_ACCUM_STEPS = 1   
EPOCHS = 15                
LEARNING_RATE = 2e-4      
MIN_LR = 1e-5
WEIGHT_DECAY = 0.0        
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

USER_TAG = "User: "
ASSISTANT_TAG = "\nAssistant: "

EOT_TOKEN = 50256  # tiktoken.get_encoding("gpt2").eot_token


# -----------------------------------------------------------------------------
# Dataset & Collate Function with Causal Shift and Loss Masking
# -----------------------------------------------------------------------------
class SFTDataset(Dataset):
    def __init__(self, jsonl_path: str, tokenizer):
        self.samples = []
        self.tokenizer = tokenizer
        self.eot_token = tokenizer.eot_token

        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                prompt = data["prompt"].strip()
                response = data["response"].strip()

                # Build the prompt prefix without a trailing space
                # Let a single newline separate header from response
                prompt_text = f"User: {prompt}\nAssistant:\n"
                full_text = f"{prompt_text}{response}"

                # Tokenize the prompt prefix and full text
                prompt_ids = tokenizer.encode(prompt_text, allowed_special={"<|endoftext|>"})
                full_ids = tokenizer.encode(full_text, allowed_special={"<|endoftext|>"}) + [self.eot_token]

                prompt_len = len(prompt_ids)
                user_ids = full_ids[:prompt_len]
                resp_ids = full_ids[prompt_len:]

                if len(full_ids) > CONTEXT_LENGTH:
                    continue

                self.samples.append((user_ids, resp_ids))

        print(f"Loaded {len(self.samples)} valid SFT samples from {jsonl_path}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def collate_fn(batch):
    input_ids_list = []
    target_ids_list = []

    # Find maximum length in THIS batch only
    max_len = max(len(u) + len(r) for u, r in batch)
    max_len = min(max_len, CONTEXT_LENGTH)

    for user_ids, resp_ids in batch:
        full_seq = (user_ids + resp_ids)[:max_len]
        full_targets = ([-100] * len(user_ids) + resp_ids)[:max_len]

        x_seq = full_seq[:-1]
        y_seq = full_targets[1:]

        # Pad only to the batch maximum, not the global 256
        pad_len = (max_len - 1) - len(x_seq)
        x_seq = x_seq + [EOT_TOKEN] * pad_len
        y_seq = y_seq + [-100] * pad_len

        input_ids_list.append(x_seq)
        target_ids_list.append(y_seq)

    x = torch.tensor(input_ids_list, dtype=torch.long)
    y = torch.tensor(target_ids_list, dtype=torch.long)
    return x, y


# -----------------------------------------------------------------------------
# Learning Rate Schedule
# -----------------------------------------------------------------------------
def get_lr(step: int, total_steps: int) -> float:
    warmup_steps = int(total_steps * 0.05)
    if step < warmup_steps:
        return LEARNING_RATE * (step + 1) / (warmup_steps + 1)
    if step > total_steps:
        return MIN_LR
    decay_ratio = (step - warmup_steps) / (total_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return MIN_LR + coeff * (LEARNING_RATE - MIN_LR)


# -----------------------------------------------------------------------------
# Main Execution
# -----------------------------------------------------------------------------
def main():
    print(f"Using device: {DEVICE}")
    enc = tiktoken.get_encoding("gpt2")

    # 1. Initialize Dataset and DataLoader
    dataset = SFTDataset(DATA_PATH, enc)
    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
        drop_last=True
    )

    # 2. Load Base Checkpoint
    print(f"Loading base checkpoint: {CHECKPOINT_IN}")
    ckpt = torch.load(CHECKPOINT_IN, map_location="cpu", weights_only=False)

    if "config" in ckpt:
        config = ckpt["config"]
    else:
        config = GPTConfig(
            vocab_size=50257,
            block_size=CONTEXT_LENGTH,
            n_layer=6,
            n_head=6,
            n_embd=384
        )

    model = GPT(config)

    # Unpack state dict by checking for known save keys
    if "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
    elif "model" in ckpt:
        state_dict = ckpt["model"]
    else:
        state_dict = ckpt

    clean_state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(clean_state_dict)

    # Explicitly re-enforce weight tying after state dict load with cast for type checkers
    wte_layer = cast(nn.Embedding, model.transformer["wte"])
    model.lm_head.weight = wte_layer.weight

    model.to(DEVICE)
    model.train()

    # 3. Setup Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        betas=(0.9, 0.95),
        weight_decay=WEIGHT_DECAY
    )

    total_steps = (len(dataloader) // GRAD_ACCUM_STEPS) * EPOCHS
    print(f"Total training steps: {total_steps} across {EPOCHS} epochs")

    step = 0
    t0 = time.time()
    running_loss = 0.0
    accum_loss = 0.0

    # 4. Training Loop
    optimizer.zero_grad()

    for epoch in range(EPOCHS):
        for it, (x, y) in enumerate(dataloader):
            x, y = x.to(DEVICE), y.to(DEVICE)

            logits, loss = model(x, targets=y)

            # Backward pass receives the scaled loss for correct gradient magnitude
            (loss / GRAD_ACCUM_STEPS).backward()

            # Store the unscaled loss value for tracking
            accum_loss += loss.item()

            if (it + 1) % GRAD_ACCUM_STEPS == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

                lr = get_lr(step, total_steps)
                for param_group in optimizer.param_groups:
                    param_group["lr"] = lr

                optimizer.step()
                optimizer.zero_grad()

                # Average the loss across the accumulation steps
                running_loss += accum_loss / GRAD_ACCUM_STEPS
                accum_loss = 0.0
                step += 1

                if step % 20 == 0 or step == total_steps:
                    dt = time.time() - t0
                    t0 = time.time()
                    interval = 20 if step % 20 == 0 else (step % 20)
                    avg_loss = running_loss / interval
                    running_loss = 0.0
                    print(
                        f"Epoch {epoch+1}/{EPOCHS} | Step {step:4d}/{total_steps} | "
                        f"Loss: {avg_loss:.4f} | LR: {lr:.2e} | "
                        f"Time: {dt:.2f}s"
                    )

    # 5. Save SFT Model
    print(f"\nTraining complete. Saving aligned model to {CHECKPOINT_OUT}...")
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": config,
        "step": step,
    }, CHECKPOINT_OUT)
    print("Done.")


if __name__ == "__main__":
    main()