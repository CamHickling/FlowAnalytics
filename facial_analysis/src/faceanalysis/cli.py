"""Command line entry point: python -m faceanalysis <command>."""

import argparse

from .config import load_config


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(prog="faceanalysis", description="Facial analysis of videos with OpenFace.")
    parser.add_argument("-c", "--config", default="config.yaml", help="Path to config file (default: config.yaml)")
    sub = parser.add_subparsers(dest="command", required=True)

    p_list = sub.add_parser("list", help="List the videos that would be processed")
    p_extract = sub.add_parser("extract", help="Run OpenFace on all videos (skips finished ones)")
    p_run = sub.add_parser("run", help="extract + analyze + plot")
    for p in (p_extract, p_run):
        p.add_argument("--limit", type=int, help="Only process the first N unfinished videos")
        p.add_argument("--force", action="store_true", help="Re-run videos that already finished")
        p.add_argument("--workers", type=int, help="Override number of parallel OpenFace processes")
    sub.add_parser("analyze", help="Quality-filter frames and write summary.csv")
    sub.add_parser("plot", help="Write an AU heatmap per video")

    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    if getattr(args, "workers", None):
        cfg.workers = args.workers

    # Imported lazily so `list` works without matplotlib etc.
    if args.command == "list":
        from .extract import done_marker, find_videos
        videos = find_videos(cfg)
        for v in videos:
            status = "done" if done_marker(cfg, v.video_id).exists() else "todo"
            print(f"[{status}] {v.video_id}  <-  {v.path}")
        print(f"{len(videos)} videos")
    if args.command in ("extract", "run"):
        from .extract import extract_all
        extract_all(cfg, limit=args.limit, force=args.force)
    if args.command in ("analyze", "run"):
        from .analyze import analyze_all
        analyze_all(cfg)
    if args.command in ("plot", "run"):
        from .plot import plot_all
        plot_all(cfg)


if __name__ == "__main__":
    main()
