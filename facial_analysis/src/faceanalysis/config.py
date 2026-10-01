from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass
class Config:
    root: Path
    openface_exe: Path
    openface_flags: list[str]
    save_tracked_video: bool
    input_dirs: list[Path]
    extensions: list[str]
    patterns: list[str] | None  # globs relative to each input dir; replaces the extension scan
    min_size_mb: float
    output_dir: Path
    workers: int
    min_confidence: float
    plot_smoothing_s: float
    min_duration_s: float
    metadata_pattern: str | None  # regex on the relative path; named groups become summary columns
    metadata_defaults: dict
    metadata_flags: dict  # flag name -> {column: value}; True when every pair matches

    @property
    def openface_dir(self) -> Path:
        return self.output_dir / "openface"

    @property
    def clean_dir(self) -> Path:
        return self.output_dir / "clean"

    @property
    def logs_dir(self) -> Path:
        return self.output_dir / "logs"

    @property
    def plots_dir(self) -> Path:
        return self.output_dir / "plots"


def load_config(path: str | Path) -> Config:
    path = Path(path).resolve()
    root = path.parent
    raw = yaml.safe_load(path.read_text())
    meta = raw.get("metadata") or {}

    def resolve(p: str) -> Path:
        p = Path(p)
        return (p if p.is_absolute() else root / p).resolve()

    return Config(
        root=root,
        openface_exe=resolve(raw["openface"]["exe"]),
        openface_flags=list(raw["openface"].get("flags", [])),
        save_tracked_video=bool(raw["openface"].get("save_tracked_video", False)),
        input_dirs=[resolve(d) for d in raw["input"]["dirs"]],
        extensions=[e.lower() for e in raw["input"].get("extensions", [])],
        patterns=raw["input"].get("patterns"),
        min_size_mb=float(raw["input"].get("min_size_mb", 0)),
        output_dir=resolve(raw["output"]["dir"]),
        workers=int(raw.get("workers", 1)),
        min_confidence=float(raw["analysis"]["min_confidence"]),
        plot_smoothing_s=float(raw["analysis"].get("plot_smoothing_s", 1.0)),
        min_duration_s=float(raw["analysis"].get("min_duration_s", 0)),
        metadata_pattern=meta.get("pattern"),
        metadata_defaults=meta.get("defaults") or {},
        metadata_flags=meta.get("flags") or {},
    )
