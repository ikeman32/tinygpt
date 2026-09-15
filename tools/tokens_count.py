#!/usr/bin/env python3
import glob
import os
import tiktoken

enc = tiktoken.get_encoding("gpt2")


def count_tokens_in_paths(patterns):
    if isinstance(patterns, str):
        patterns = [patterns]

    total_tokens = 0
    matched_files = set()

    for pattern in patterns:
        for path in glob.glob(pattern, recursive=True):
            if os.path.isfile(path):
                matched_files.add(os.path.abspath(path))

    for file_path in matched_files:
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()
            total_tokens += len(enc.encode(text, allowed_special={"<|endoftext|>"}))
        except Exception:
            pass

    return len(matched_files), total_tokens


# Pass a list of patterns for C files to capture both extensions
gutenberg_files, gutenberg_tokens = count_tokens_in_paths("data/**/gutenberg*/**/*.txt")
c_files, c_tokens = count_tokens_in_paths([
    "data/**/c_source*/**/*.c",
    "data/**/c_source*/**/*.h",
])
py_files, py_tokens = count_tokens_in_paths("data/**/py_source*/**/*.py")
legal_files, legal_tokens = count_tokens_in_paths("data/**/legal*/**/*.txt")
py_doc_files, py_doc_tokens = count_tokens_in_paths("data/**/py_docs*/**/*.txt")
speech_files, speech_tokens = count_tokens_in_paths("data/**/speeches*/**/*.txt")
man_files, man_tokens = count_tokens_in_paths("data/**/manpages*/**/*.txt")

print(f"Gutenberg:       {gutenberg_files:>5} files | {gutenberg_tokens:>12,} tokens")
print(f"Code (C):        {c_files:>5} files | {c_tokens:>12,} tokens")
print(f"Code (Python):   {py_files:>5} files | {py_tokens:>12,} tokens")
print(f"Text (Legal):    {legal_files:>5} files | {legal_tokens:>12,} tokens")
print(f"Text (Py Docs):  {py_doc_files:>5} files | {py_doc_tokens:>12,} tokens")
print(f"Text (Speeches): {speech_files:>5} files | {speech_tokens:>12,} tokens")
print(f"Text (Manpages): {man_files:>5} files | {man_tokens:>12,} tokens")

total_all_tokens = (
    gutenberg_tokens + c_tokens + py_tokens + legal_tokens + py_doc_tokens + speech_tokens + man_tokens
)
print("-" * 45)
print(f"Total counted:             {total_all_tokens:>12,} tokens")