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
    p_trials = sub.add_parser("trials", help="FlowAnalytics trial face videos synced to the performance playback")
    p_trials.add_argument("action", choices=["list", "extract", "analyze", "plot", "run"])
    p_trials.add_argument("--only", nargs="+", help="Trial folder prefixes to process, e.g. P02B P20")
    p_trials.add_argument("--limit", type=int)
    p_trials.add_argument("--force", action="store_true")
    p_trials.add_argument("--workers", type=int)

    args = parser.parse_args(argv)
    if args.command == "trials":
        return run_trials(args)
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


def run_trials(args) -> None:
    from . import trials
    cfg, settings = trials.load_settings(args.config)
    if args.workers:
        cfg.workers = args.workers
    if args.action == "list":
        videos = trials.find_trial_videos(settings)
        for tv in videos:
            if tv.face_video is None:
                print(f"[missing] {tv.video_id}")
                continue
            w = trials.window_for(tv, settings)
            pauses = sum(s.state == "paused" for s in w.segments)
            print(f"{tv.video_id:40} start {w.face_start_s or 0:7.1f}s  length {w.duration_s or 0:6.1f}s  "
                  f"pauses {pauses:2}  {'; '.join(w.flags)}")
        print(f"{sum(tv.face_video is not None for tv in videos)} of {len(videos)} trial/condition videos found")
    if args.action in ("extract", "run"):
        trials.extract_trials(cfg, settings, limit=args.limit, only=args.only, force=args.force)
    if args.action in ("analyze", "run"):
        trials.analyze_trials(cfg, settings)
    if args.action in ("plot", "run"):
        trials.plot_trials(cfg)


if __name__ == "__main__":
    main()
