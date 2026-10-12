#!/usr/bin/env python3
"""
codepack.py - AI Context Packager & Codebase Documenter
Recursively crawls a project tree and bundles codebase structure + file contents
into a single LLM-optimized Markdown document.
"""

import os
import sys
import fnmatch
import argparse
from pathlib import Path

DEFAULT_IGNORE_DIRS = {
    '.git', '.svn', '.hg', 'node_modules', '__pycache__', '.pytest_cache',
    '.mypy_cache', '.venv', 'venv', 'env', '.idea', '.vscode', 'dist', 
    'build', '.next', '.nuxt', 'coverage', '.eggs', '*.egg-info'
}

VIDEO_EXTENSIONS = {
    '.mp4', '.mov', '.avi', '.mkv', '.m4v', '.wmv', '.flv', '.webm', '.3gp'
}

NON_TEXT_EXTENSIONS = {
    # Video
    *VIDEO_EXTENSIONS,
    # Audio
    '.mp3', '.wav', '.aac', '.flac', '.ogg', '.m4a',
    # Images
    '.png', '.jpg', '.jpeg', '.gif', '.ico', '.webp', '.tiff', '.bmp',
    # Archives / Binaries / Data dumps
    '.zip', '.tar', '.gz', '.7z', '.rar', '.pdf', '.exe', '.dll', '.so', 
    '.dylib', '.bin', '.pkl', '.npy', '.parquet', '.sqlite', '.db', 
    '.pyc', '.pyo', '.woff', '.woff2', '.ttf', '.eot'
}

def format_size(size_bytes: int) -> str:
    """Format bytes into human-readable size."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}" if unit != 'B' else f"{size_bytes} B"
        size_bytes /= 1024
    return f"{size_bytes:.1f} PB"

def is_binary_string(bytes_data: bytes) -> bool:
    """Detect if file content contains null bytes."""
    return b'\x00' in bytes_data

def load_gitignore(root_dir: Path) -> list[str]:
    """Parse .gitignore file if it exists."""
    gitignore_path = root_dir / ".gitignore"
    patterns = []
    if gitignore_path.exists():
        with open(gitignore_path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    patterns.append(line)
    return patterns

def matches_ignore(rel_path: str, patterns: list[str]) -> bool:
    """Check if relative path matches any ignore patterns."""
    parts = rel_path.split(os.sep)
    for part in parts:
        if part in DEFAULT_IGNORE_DIRS:
            return True
        for pattern in patterns:
            if fnmatch.fnmatch(part, pattern.rstrip('/')):
                return True
            if fnmatch.fnmatch(rel_path, pattern):
                return True
    return False

def build_tree(root_dir: Path, ignore_patterns: list[str]) -> tuple[str, list[Path], list[Path]]:
    """Build ASCII tree and return lists of text files and video files."""
    tree_lines = [f"{root_dir.resolve().name}/"]
    text_files = []
    video_files = []

    def _walk(directory: Path, prefix: str = ""):
        try:
            entries = sorted(list(directory.iterdir()), key=lambda e: (not e.is_dir(), e.name.lower()))
        except PermissionError:
            return

        # Filter out ignored directories and general non-video binaries
        filtered_entries = []
        for e in entries:
            rel = str(e.relative_to(root_dir))
            if matches_ignore(rel, ignore_patterns):
                continue
            
            # Keep directories, text files, and video files in the tree
            if e.is_file():
                ext = e.suffix.lower()
                if ext in VIDEO_EXTENSIONS:
                    filtered_entries.append(e)
                elif ext not in NON_TEXT_EXTENSIONS:
                    filtered_entries.append(e)
            else:
                filtered_entries.append(e)

        for i, entry in enumerate(filtered_entries):
            is_last = (i == len(filtered_entries) - 1)
            connector = "└── " if is_last else "├── "

            if entry.is_dir():
                tree_lines.append(f"{prefix}{connector}{entry.name}/")
                extension = "    " if is_last else "│   "
                _walk(entry, prefix + extension)
            else:
                ext = entry.suffix.lower()
                if ext in VIDEO_EXTENSIONS:
                    try:
                        size_str = f" [{format_size(entry.stat().st_size)}]"
                    except OSError:
                        size_str = ""
                    tree_lines.append(f"{prefix}{connector}{entry.name}{size_str} (video)")
                    video_files.append(entry)
                else:
                    tree_lines.append(f"{prefix}{connector}{entry.name}")
                    text_files.append(entry)

    _walk(root_dir)
    return "\n".join(tree_lines), text_files, video_files

def generate_context_doc(root_path: Path, output_file: Path, user_instructions: str = None):
    ignore_patterns = load_gitignore(root_path)
    tree_view, text_files, video_files = build_tree(root_path, ignore_patterns)

    print(f"Discovered {len(text_files)} text/code files and {len(video_files)} video files.")

    with open(output_file, "w", encoding="utf-8") as out:
        out.write("# Codebase Context & Modification Request\n\n")
        
        if user_instructions:
            out.write("## 🎯 Goal / Planned Changes\n")
            out.write(f"{user_instructions}\n\n")
            out.write("---\n\n")
            
        out.write("## 📁 Directory Structure\n")
        out.write("```text\n")
        out.write(tree_view)
        out.write("\n```\n\n")

        if video_files:
            out.write("### 🎬 Media Files Identified\n")
            out.write("| Relative Path | Size |\n|---|---|\n")
            for vf in video_files:
                rel = vf.relative_to(root_path)
                try:
                    size = format_size(vf.stat().st_size)
                except OSError:
                    size = "Unknown"
                out.write(f"| `{rel}` | {size} |\n")
            out.write("\n")

        out.write("---\n\n")
        out.write("## 📄 File Contents\n\n")

        for file_path in text_files:
            rel_path = file_path.relative_to(root_path)
            
            try:
                raw_data = file_path.read_bytes()
                if is_binary_string(raw_data[:1024]):
                    continue
                content = raw_data.decode("utf-8", errors="replace")
            except Exception as e:
                content = f"[Error reading file: {e}]"

            ext = file_path.suffix.lstrip(".")
            lang = ext if ext else "text"

            out.write(f"### File: `{rel_path}`\n")
            out.write(f"```{lang}\n")
            out.write(content.rstrip())
            out.write("\n```\n\n")

    print(f"✓ Context bundle written to: {output_file.resolve()}")

def main():
    parser = argparse.ArgumentParser(
        description="Recursively pack repository structure and text files, showing video filenames in the tree."
    )
    parser.add_argument("path", nargs="?", default=".", help="Root directory (default: .)")
    parser.add_argument("-o", "--output", default="ai_code_context.md", help="Output file")
    parser.add_argument("-g", "--goal", default=None, help="Prompt/goal explaining what to change")

    args = parser.parse_args()
    root = Path(args.path).resolve()

    if not root.is_dir():
        print(f"Error: Directory '{root}' does not exist.", file=sys.stderr)
        sys.exit(1)

    generate_context_doc(root, Path(args.output), args.goal)

if __name__ == "__main__":
    main()