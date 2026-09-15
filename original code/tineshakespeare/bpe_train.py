import os
import json
import math
import torch
import tiktoken
from tqdm import tqdm
from model import GPTConfig, GPT
from generate import generate


def load_data(filepath, block_size, batch_size, device):
    with open(filepath, "r", encoding="utf-8") as f:
        text = f.read()

    # Use standard GPT-2 byte pair encoding (vocab size = 50,257)
    enc = tiktoken.get_encoding("gpt2")
    tokens = torch.tensor(enc.encode(text, allowed_special={"<|endoftext|>"}), dtype=torch.long)
    vocab_size = enc.n_vocab

    print(f"Dataset: {len(tokens):,} BPE tokens, vocab size: {vocab_size:,}")

    def get_batch(split_tokens):
        ix = torch.randint(len(split_tokens) - block_size - 1, (batch_size,))
        x = torch.stack([split_tokens[i:i + block_size] for i in ix]).to(device)
        y = torch.stack([split_tokens[i + 1:i + block_size + 1] for i in ix]).to(device)
        return x, y

    n = int(0.9 * len(tokens))
    get_train = lambda: get_batch(tokens[:n])
    get_val = lambda: get_batch(tokens[n:])
    return get_train, get_val, vocab_size, enc


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")     # Apple Silicon GPU
    elif torch.cuda.is_available():
        return torch.device("cuda")    # NVIDIA GPU
    return torch.device("cpu")


def get_lr(step, warmup_steps, max_steps, max_lr, min_lr):
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps
    if step >= max_steps:
        return min_lr
    progress = (step - warmup_steps) / (max_steps - warmup_steps)
    return min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * progress))


def train(data_path, checkpoint_path=None, max_steps=5000, batch_size=16,
          n_layer=6, n_head=6, n_embd=384, block_size=256):
    device = get_device()
    print(f"Using device: {device}")

    get_train_batch, get_val_batch, vocab_size, enc = load_data(
        data_path, block_size, batch_size, device
    )

    config = GPTConfig(
        vocab_size=vocab_size,
        block_size=block_size,
        n_layer=n_layer,
        n_head=n_head,
        n_embd=n_embd,
    )
    model = GPT(config).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)

    start_step = 0
    loss_log = {"steps": [], "train": [], "val": []}
    best_val_loss = float("inf")

    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])

        if "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        start_step = checkpoint.get("step", 0)
        best_val_loss = checkpoint.get("best_val_loss", float("inf"))

        if os.path.exists("loss_log.json"):
            with open("loss_log.json", "r") as f:
                loss_log = json.load(f)

        print(f"Resuming training from step {start_step} to {max_steps} (best val loss: {best_val_loss:.4f})")
    else:
        print("Starting training from scratch")

    print(f"Model: {n_layer}L/{n_head}H/{n_embd}D, "
          f"{sum(p.numel() for p in model.parameters()) / 1e6:.1f}M params")

    max_lr = 1e-3
    min_lr = max_lr * 0.1
    warmup_steps = 100

    # Ensure val_loss is bound prior to loop entry
    val_loss = float("inf")

    pbar = tqdm(range(start_step, max_steps), initial=start_step, total=max_steps, desc="Training")
    for step in pbar:
        # --- validation loss ---
        if step % 100 == 0:
            model.eval()
            with torch.no_grad():
                val_losses = []
                for _ in range(5):  # Reduced from 20 to 5 batches to conserve CPU RAM
                    x, y = get_val_batch()
                    _, loss = model(x, y)
                    val_losses.append(loss.item())
                val_loss = sum(val_losses) / len(val_losses)
                tqdm.write(f"Step {step:5d} | val loss: {val_loss:.4f}")

                # Log validation loss directly in scope
                loss_log["val"].append(val_loss)

                # Save best checkpoint
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    tqdm.write(f"--> Step {step:5d}: New lowest val loss ({best_val_loss:.4f}). Saving checkpoint_best.pt")
                    torch.save({
                        "step": step,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "config": config,
                        "best_val_loss": best_val_loss,
                        "tokenizer": "gpt2",
                    }, "checkpoint_best.pt")

            model.train()

        # --- update learning rate ---
        lr = get_lr(step, warmup_steps, max_steps, max_lr, min_lr)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        # --- training step ---
        x, y = get_train_batch()
        _, loss = model(x, y)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        pbar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{lr:.2e}")

        # --- log train loss ---
        loss_log["steps"].append(step)
        loss_log["train"].append(loss.item())

        # --- generate sample (BPE token tensor) ---
        if step > 0 and step % 100 == 0:
            model.eval()
            prompt = "To be or not"
            prompt_tokens = torch.tensor(enc.encode(prompt), dtype=torch.long, device=device).unsqueeze(0)
            
            with torch.no_grad():
                out_tokens = generate(model, prompt_tokens, max_new_tokens=50, temperature=0.8)
                sample = enc.decode(out_tokens[0].tolist())
            tqdm.write(f"\n--- Step {step} sample ---\n{sample}\n---\n")
            model.train()

        # --- save periodic checkpoint ---
        if step > 0 and step % 1000 == 0:
            torch.save({
                "step": step,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "config": config,
                "best_val_loss": best_val_loss,
                "tokenizer": "gpt2",
            }, f"checkpoint_{step}.pt")

    # --- save final checkpoint and loss log ---
    torch.save({
        "step": max_steps,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": config,
        "best_val_loss": best_val_loss,
        "tokenizer": "gpt2",
    }, "checkpoint_final.pt")

    with open("loss_log.json", "w") as f:
        json.dump(loss_log, f)

    return model, enc


if __name__ == "__main__":
    import sys
    # Usage: uv run train.py <data_path> <checkpoint_path> <max_steps>
    data_path = sys.argv[1] if len(sys.argv) > 1 else "./data/shakespeare.txt"
    checkpoint_path = sys.argv[2] if len(sys.argv) > 2 and sys.argv[2] != "None" else None
    max_steps = int(sys.argv[3]) if len(sys.argv) > 3 else 5000

    train(data_path=data_path, checkpoint_path=checkpoint_path, max_steps=max_steps)