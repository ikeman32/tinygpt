#!/usr/bin/env python3
import os
import sys
from pathlib import Path

# Target directory to process
TARGET_DIR = "./data/corpus/c_source"

# Supported file extensions and comment formats
EXT_CONFIG = {
    ".py": {
        "language": "Python",
        "header_fmt": "# File: {filename}\n# Language: Python\n\n"
    },
    ".c": {
        "language": "C",
        "header_fmt": "// File: {filename}\n// Language: C\n\n"
    },
    ".h": {
        "language": "C",
        "header_fmt": "// File: {filename}\n// Language: C\n\n"
    },
}


def prepend_header(file_path: Path, root_path: Path) -> bool:
    config = EXT_CONFIG.get(file_path.suffix)
    if not config:
        return False

    # Store relative path so module/package hierarchy is preserved
    relative_path = file_path.relative_to(root_path).as_posix()
    header = config["header_fmt"].format(filename=relative_path)

    try:
        content = file_path.read_text(encoding="utf-8", errors="replace")

        # Avoid duplicate prepending if run multiple times
        first_line = content.splitlines()[0] if content else ""
        if "File:" in first_line or "File :" in first_line:
            return False

        # Preserve shebang on line 1 for scripts if present
        if content.startswith("#!"):
            lines = content.split("\n", 1)
            shebang = lines[0] + "\n"
            rest = lines[1] if len(lines) > 1 else ""
            new_content = shebang + header + rest
        else:
            new_content = header + content

        file_path.write_text(new_content, encoding="utf-8")
        return True

    except Exception as e:
        print(f"Error processing {file_path}: {e}", file=sys.stderr)
        return False


def main():
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(TARGET_DIR)

    if not target.exists():
        print(f"Target path '{target}' does not exist.")
        sys.exit(1)

    count = 0
    skipped = 0

    for root, _, files in os.walk(target):
        root_path = Path(root)
        for file in files:
            file_path = root_path / file
            if file_path.suffix in EXT_CONFIG:
                if prepend_header(file_path, target):
                    count += 1
                else:
                    skipped += 1

    print(f"Updated {count} files (skipped {skipped} already tagged or unsupported).")


if __name__ == "__main__":
    main()
