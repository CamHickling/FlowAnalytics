#!/usr/bin/env python3
import os
from pathlib import Path


def get_session_names(base_dir: Path) -> list[str]:
    """Return sorted session names from the given base directory."""
    if not base_dir.exists():
        raise FileNotFoundError(f"Base directory does not exist: {base_dir}")
    if not base_dir.is_dir():
        raise NotADirectoryError(f"Base path is not a directory: {base_dir}")

    session_names = [entry.name for entry in sorted(base_dir.iterdir()) if entry.is_dir()]
    return session_names


if __name__ == "__main__":
    root = Path(__file__).resolve().parent
    sessions_dir = root / "Iris_Recorded_Taekwondo_Data"

    try:
        session_names = get_session_names(sessions_dir)
    except (FileNotFoundError, NotADirectoryError) as exc:
        print(exc)
        raise SystemExit(1)

    if not session_names:
        print("No sessions found in:", sessions_dir)
    else:
        print("Found session names:")
        for name in session_names:
            print("-", name)
