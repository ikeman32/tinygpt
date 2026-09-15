import json
import math
import os
import sys
import torch
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from dataset import RawTextDirectoryDataset
from generate import generate
from model import GPT, GPTConfig


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def get_lr(step, warmup_steps, max_steps, max_lr, min_lr):
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps
    if step >= max_steps:
        return min_lr
    progress = (step - warmup_steps) / (max_steps - warmup_steps)
    return min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * progress))


def cycle(iterable):
    """Infinite iterator over DataLoader batches across epochs."""
    while True:
        for x in iterable:
            yield x


def train(data_path, checkpoint_path=None, max_steps=5000, batch_size=16,
          n_layer=6, n_head=6, n_embd=384, block_size=256):
    device = get_device()
    print(f"Using device: {device}")

    # Instantiate the exact class from your dataset.py
    full_dataset = RawTextDirectoryDataset(data_dir=data_path, block_size=block_size)
    enc = full_dataset.enc
    vocab_size = enc.n_vocab

    val_size = max(1, int(len(full_dataset) * 0.1))
    train_size = len(full_dataset) - val_size

    train_data, val_data = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(1337)
    )

    print(f"Dataset split: {len(train_data):,} train chunks, {len(val_data):,} val chunks")

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=True, drop_last=False)

    train_iter = cycle(train_loader)
    val_iter = cycle(val_loader)

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

    pbar = tqdm(range(start_step, max_steps), initial=start_step, total=max_steps, desc="Training")
    for step in pbar:
        # --- validation loss ---
        if step % 100 == 0:
            model.eval()
            with torch.no_grad():
                val_losses = []
                for _ in range(5):
                    x, y = next(val_iter)
                    x, y = x.to(device), y.to(device)
                    _, loss = model(x, y)
                    val_losses.append(loss.item())
                val_loss = sum(val_losses) / len(val_losses)
                tqdm.write(f"Step {step:5d} | val loss: {val_loss:.4f}")

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
        x, y = next(train_iter)
        x, y = x.to(device), y.to(device)

        _, loss = model(x, y)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        pbar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{lr:.2e}")

        # --- log train loss ---
        loss_log["steps"].append(step)
        loss_log["train"].append(loss.item())

        # --- generate sample ---
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
    # Allows passing multiple folders separated by commas or spaces
    # Example: uv run train.py ./data/en/gutenberg,./data/source_code
    if len(sys.argv) > 1 and sys.argv[1] != "None":
        if "," in sys.argv[1]:
            data_path = [p.strip() for p in sys.argv[1].split(",")]
        else:
            data_path = sys.argv[1]
    else:
        # Default fallback to your project's top-level data folder
        data_path = "./data"

    checkpoint_path = sys.argv[2] if len(sys.argv) > 2 and sys.argv[2] != "None" else None
    max_steps = int(sys.argv[3]) if len(sys.argv) > 3 else 5000

    train(data_path=data_path, checkpoint_path=checkpoint_path, max_steps=max_steps)