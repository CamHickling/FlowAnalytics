"""Per-video AU intensity heatmaps."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from tqdm import tqdm

from .analyze import AU_INTENSITY
from .config import Config

AU_NAMES = {
    "AU01": "Inner brow raiser", "AU02": "Outer brow raiser", "AU04": "Brow lowerer",
    "AU05": "Upper lid raiser", "AU06": "Cheek raiser", "AU07": "Lid tightener",
    "AU09": "Nose wrinkler", "AU10": "Upper lip raiser", "AU12": "Lip corner puller",
    "AU14": "Dimpler", "AU15": "Lip corner depressor", "AU17": "Chin raiser",
    "AU20": "Lip stretcher", "AU23": "Lip tightener", "AU25": "Lips part",
    "AU26": "Jaw drop", "AU45": "Blink",
}

# Single-hue sequential ramp (light -> dark blue); near-zero recedes into the white surface.
INTENSITY_CMAP = LinearSegmentedColormap.from_list(
    "au_intensity", ["#ffffff", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]
)
INTENSITY_CMAP.set_bad("#d9d8d4")  # neutral gray = face not tracked
INK, MUTED = "#1f1f1e", "#6b6a66"


def plot_video(clean: pd.DataFrame, video_id: str, smoothing_s: float, out_path) -> None:
    au_cols = list(clean.filter(regex=AU_INTENSITY).columns)
    fps = 1 / np.median(np.diff(clean["timestamp"])) if len(clean) > 1 else 30
    window = max(1, int(round(smoothing_s * fps)))
    smooth = clean[au_cols].rolling(window, center=True, min_periods=max(1, window // 2)).mean()
    smooth[~clean["valid"]] = np.nan  # don't let smoothing paint over untracked stretches

    t0, t1 = clean["timestamp"].iloc[0], clean["timestamp"].iloc[-1]
    minutes = (t1 - t0) > 120
    scale = 60 if minutes else 1

    fig, ax = plt.subplots(figsize=(12, 5.5), constrained_layout=True)
    im = ax.imshow(
        smooth.T.to_numpy(), aspect="auto", interpolation="nearest", cmap=INTENSITY_CMAP,
        vmin=0, vmax=5, extent=[t0 / scale, t1 / scale, len(au_cols) - 0.5, -0.5],
    )
    ax.set_yticks(range(len(au_cols)))
    ax.set_yticklabels([f"{c[:4]}  {AU_NAMES.get(c[:4], '')}" for c in au_cols], fontsize=8, color=INK)
    ax.set_xlabel(f"Time ({'min' if minutes else 's'})", color=MUTED)
    ax.tick_params(colors=MUTED, length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)

    valid_pct = clean["valid"].mean()
    ax.set_title(
        f"{video_id}\nAU intensity ({smoothing_s:g}s rolling mean) · face tracked in {valid_pct:.0%} of frames "
        f"· gray = not tracked",
        loc="left", fontsize=10, color=INK,
    )
    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01)
    cbar.set_label("Intensity (0-5)", color=MUTED)
    cbar.outline.set_visible(False)
    cbar.ax.tick_params(colors=MUTED, length=0)

    fig.savefig(out_path, dpi=130, facecolor="white")
    plt.close(fig)


def plot_all(cfg: Config) -> None:
    cfg.plots_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(cfg.clean_dir.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No cleaned outputs in {cfg.clean_dir}. Run `analyze` first.")
    for path in tqdm(files, unit="plot"):
        clean = pd.read_csv(path)
        plot_video(clean, path.stem, cfg.plot_smoothing_s, cfg.plots_dir / f"{path.stem}.png")
    print(f"Wrote {len(files)} plot(s) to {cfg.plots_dir}")
