#!/usr/bin/env python3
import os
import re

# Prompt with clean spacing and default fallbacks
root_in = input("Enter root directory path (e.g. ./man_pages_raw): ").strip()
root_dir = root_in if root_in else "./man_pages_raw"

corpus_in = input("Enter name of the corpus file (e.g. man_corpus): ").strip()
corpus_name = corpus_in if corpus_in else "corpus"

# Strip .txt extension if the user typed it in manually
if corpus_name.endswith(".txt"):
    corpus_name = corpus_name[:-4]

output_dir = "./data"
os.makedirs(output_dir, exist_ok=True)
output_file = os.path.join(output_dir, f"{corpus_name}.txt")

boundary = "\n<|endoftext|>\n"

# Regex to strip ANSI escape sequences
ansi_pat = re.compile(r"\x1b\[[0-9;]*[a-zA-Z]")
# Regex to match character + backspace pairs
backspace_pat = re.compile(r".\x08")

total_files = 0
total_chars = 0

print(f"Scanning '{root_dir}' and compiling into '{output_file}'...")

with open(output_file, "w", encoding="utf-8") as out_f:
    for dirpath, _, filenames in os.walk(root_dir):
        for fname in sorted(filenames):
            fpath = os.path.join(dirpath, fname)
            
            # Skip symlinks
            if os.path.islink(fpath):
                continue

            try:
                with open(fpath, "r", encoding="utf-8", errors="ignore") as in_f:
                    raw_text = in_f.read()

                # Clean escape sequences first
                clean_text = ansi_pat.sub("", raw_text)
                
                # Iteratively remove backspaces until none remain
                while "\x08" in clean_text:
                    clean_text = backspace_pat.sub("", clean_text)
                
                clean_text = clean_text.strip()

                # Skip trivial stubs, redirect pointers, or nearly empty pages
                if len(clean_text) < 150:
                    continue

                out_f.write(clean_text)
                out_f.write(boundary)
                
                total_files += 1
                total_chars += len(clean_text)

            except Exception as e:
                print(f"Skipping {fpath}: {e}")

print(f"\nDone.")
print(f"Processed: {total_files} files.")
print(f"Total corpus size: {total_chars / (1024 * 1024):.2f} MB")