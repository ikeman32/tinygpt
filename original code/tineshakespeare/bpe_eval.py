import tiktoken
import torch
from model import GPT, GPTConfig

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CHECKPOINT_PATH = "checkpoint_best.pt"

# 1. Load Checkpoint
checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=False)

# 2. Setup Tokenizer (BPE gpt2)
enc = tiktoken.get_encoding("gpt2")
encode = enc.encode
decode = enc.decode

# 3. Instantiate Model & Load Weights
raw_config = checkpoint["config"]
if isinstance(raw_config, dict):
    config = GPTConfig(**raw_config)
else:
    config = raw_config

model = GPT(config)

# Strip any compile / DDP prefixes
raw_state_dict = checkpoint["model_state_dict"]
cleaned_state_dict = {
    k.replace("_orig_mod.", "").replace("module.", ""): v
    for k, v in raw_state_dict.items()
}

model.load_state_dict(cleaned_state_dict)
model.to(DEVICE)
model.eval()


# 4. Generation Function
@torch.no_grad()
def generate(
    prompt: str,
    max_new_tokens: int = 150,
    temperature: float = 0.8,
    top_k: int = 40,
) -> str:
    tokens = encode(prompt)
    if not tokens:
        tokens = [enc.eot_token]

    x = torch.tensor(tokens, dtype=torch.long, device=DEVICE).unsqueeze(0)

    for _ in range(max_new_tokens):
        x_cond = x if x.size(1) <= config.block_size else x[:, -config.block_size :]
        logits, _ = model(x_cond)
        logits = logits[:, -1, :] / max(temperature, 1e-5)

        if top_k is not None and top_k > 0:
            k = min(top_k, logits.size(-1))
            v, _ = torch.topk(logits, k)
            logits[logits < v[:, [-1]]] = -float("Inf")

        probs = torch.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1)
        x = torch.cat((x, idx_next), dim=1)

    return decode(x[0].tolist())


# 5. Execution & Interactive Prompt Loop
if __name__ == "__main__":
    best_loss = checkpoint.get("best_val_loss", "N/A")
    step = checkpoint.get("step", "N/A")
    print(f"Loaded {CHECKPOINT_PATH} (Step: {step}, Best Val Loss: {best_loss}) on {DEVICE.upper()}")
    print("-" * 60)

    preset_prompts = [
        "To be or not",
        "ROMEO:\nWhat light through",
        "KING RICHARD II:\n",
        "WARWICK:\n",
    ]

    for p in preset_prompts:
        print(f"\n[Prompt]:\n{p}")
        out = generate(p, max_new_tokens=120, temperature=0.75, top_k=40)
        print(f"[Generation]:\n{out}")
        print("-" * 60)

    print("\nEntering interactive mode (Ctrl+C to exit).")
    while True:
        try:
            user_prompt = input("\nEnter prompt > ")
            if not user_prompt.strip():
                continue
            res = generate(user_prompt, max_new_tokens=150, temperature=0.8, top_k=40)
            print(f"\n{res}")
        except KeyboardInterrupt:
            print("\nExiting.")
            break