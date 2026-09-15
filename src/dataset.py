#!/usr/bin/env python3
import os
from pathlib import Path
import tiktoken
import torch
from torch.utils.data import Dataset

VALID_EXTS = {".txt", ".py", ".c", ".h", ".md"}

class RawTextDirectoryDataset(Dataset):
    def __init__(self, data_dir, block_size=256, encoding_name="gpt2"):
        self.block_size = block_size
        self.enc = tiktoken.get_encoding(encoding_name)
        
        # Normalize input to a list of directories
        if isinstance(data_dir, (str, Path)):
            search_dirs = [Path(data_dir)]
        else:
            search_dirs = [Path(p) for p in data_dir]

        # Gather all matching files across all passed directories recursively
        file_paths = []
        for s_dir in search_dirs:
            if not s_dir.exists():
                print(f"Warning: Directory '{s_dir}' not found. Skipping.")
                continue
            for root, _, files in os.walk(s_dir):
                for f in files:
                    if Path(f).suffix.lower() in VALID_EXTS:
                        file_paths.append(os.path.join(root, f))

        print(f"Discovered {len(file_paths)} files across {len(search_dirs)} search paths.")

        # Read, tokenize, and chunk into fixed block_size windows
        all_tokens = []
        eot_token = self.enc.eot_token  # <|endoftext|> separator

        for path in file_paths:
            try:
                with open(path, "r", encoding="utf-8", errors="replace") as f:
                    text = f.read()
                if not text.strip():
                    continue
                tokens = self.enc.encode(text, allowed_special={"<|endoftext|>"})
                all_tokens.extend(tokens)
                all_tokens.append(eot_token)
            except Exception as e:
                print(f"Skipping {path} due to read error: {e}")

        self.tokens = torch.tensor(all_tokens, dtype=torch.long)
        num_chunks = len(self.tokens) // (self.block_size + 1)
        print(f"Total tokens: {len(self.tokens):,} | Chunks: {num_chunks:,}")

    def __len__(self):
        # We need (block_size + 1) tokens per sample for (x, y) input/target pairs
        return max(0, len(self.tokens) // (self.block_size + 1))

    def __getitem__(self, idx):
        start = idx * (self.block_size + 1)
        chunk = self.tokens[start : start + self.block_size + 1]
        x = chunk[:-1]
        y = chunk[1:]
        return x, y