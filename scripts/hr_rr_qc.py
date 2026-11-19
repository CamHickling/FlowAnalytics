#!/usr/bin/env python3
"""Heart-rate quality check from the strap's beat-to-beat (RR) intervals.

The strap's `bpm` column is its own smoothed estimate, which it keeps reporting
when it stops detecting beats -- a frozen value that looks like normal data
(P42 held 147-149 bpm for ~28 s mid-form while sensor_contact stayed True).
The RR intervals are the beats it actually measured, so they show when the
signal is real.

Each RR is accepted if it is within RR_MIN_MS..RR_MAX_MS and within RR_JUMP of
the median of the last accepted ones (rejecting missed or extra beats, e.g. a
737 ms interval that is two 368 ms beats). Each 1 Hz row is classed as:
  ok      at least one accepted RR interval
  zero    bpm reported as 0
  no_rr   bpm reported but no plausible RR interval behind it

A run of RUN_MIN_SEC or more consecutive zero/no_rr rows counts as a dropout:
single rows without RR happen normally when a beat falls across two
notifications, and runs separated by BRIDGE_ROWS or fewer ok rows are merged.
rr_bpm (60000 / median accepted RR) is written for every ok row so
analyses can use it in place of the strap's bpm.

Usage:
    python hr_rr_qc.py [--data-root DIR] [--out DIR]
Writes hr_rr_qc_sessions.csv (one row per session and phase window),
hr_rr_qc_dropouts.csv (one row per dropout run) and, per trial,
heart_rate/<pid>_hr_rr_clean.csv (every row with its class and rr_bpm).
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

RR_MIN_MS, RR_MAX_MS = 250.0, 2000.0   # 240..30 bpm
RR_JUMP = 0.30                         # reject an RR this far off the recent median (missed/extra beat)
RR_RESET_SEC = 5.0                     # no accepted RR this long: trust the next ones afresh
RUN_MIN_SEC = 3                        # consecutive bad rows that make a dropout
BRIDGE_ROWS = 2                        # isolated ok rows inside a dropout don't end it
BPM_DISAGREE = 15                      # strap bpm vs rr_bpm difference worth flagging
PRE_WINDOW_SEC = 60.0                  # pre-performance baseline window
POST_WINDOW_SEC = 15.0                 # peak window extension after the end marker

DATA_ROOT = Path(__file__).resolve().parents[1] / "Iris_Recorded_Taekwondo_Data"


class RRFilter:
    """Accepts RR intervals near the running median of recently accepted ones."""

    def __init__(self) -> None:
        self.recent: list[float] = []
        self.last_ok_time = None

    def accept(self, timestamp: float, values: list[float]) -> list[float]:
        if self.last_ok_time is not None and timestamp - self.last_ok_time > RR_RESET_SEC:
            self.recent = []
        kept = []
        for value in values:
            if not RR_MIN_MS <= value <= RR_MAX_MS:
                continue
            ref = sorted(self.recent)[len(self.recent) // 2] if len(self.recent) >= 3 else None
            if ref is None or abs(value - ref) <= RR_JUMP * ref:
                kept.append(value)
                self.recent = (self.recent + [value])[-15:]
        if kept:
            self.last_ok_time = timestamp
        return kept


def classify(row: dict, rr_filter: RRFilter) -> tuple[str, float | None]:
    values = []
    for part in (row.get("rr_intervals_ms") or "").split(";"):
        try:
            values.append(float(part))
        except ValueError:
            continue
    bpm = int(float(row["bpm"] or 0))
    kept = rr_filter.accept(float(row["timestamp"]), values)
    if bpm == 0:
        return "zero", None
    if not kept:
        return "no_rr", None
    return "ok", 60000.0 / sorted(kept)[len(kept) // 2]


def performance_window(events: list[dict]) -> tuple[float, float] | None:
    """The last overhead take: a redone performance logs several, and the last
    is the one played back in review."""
    starts = [e["wall_time"] for e in events if e["event"] == "overhead_recorder_start"]
    stops = [e["wall_time"] for e in events if e["event"] == "overhead_recorder_stop"]
    return (starts[-1], stops[-1]) if starts and stops else None


def runs(rows: list[dict]) -> list[tuple[int, int]]:
    """(first, last) index of each dropout: RUN_MIN_SEC+ bad rows, bridging
    over up to BRIDGE_ROWS isolated ok rows."""
    out, start, last_bad, good = [], None, None, 0
    for i, r in enumerate(rows + [{"cls": "ok"}] * (BRIDGE_ROWS + 1)):
        if r["cls"] != "ok":
            start = i if start is None else start
            last_bad, good = i, 0
        elif start is not None:
            good += 1
            if good > BRIDGE_ROWS:
                if last_bad - start + 1 >= RUN_MIN_SEC:
                    out.append((start, last_bad))
                start, good = None, 0
    return out


def summarize(label: str, rows: list[dict]) -> dict:
    n = len(rows)
    ok = [r for r in rows if r["cls"] == "ok"]
    bad = n - len(ok)
    run_list = runs(rows)
    strap = [r["bpm"] for r in rows if r["bpm"] > 0]
    rr = [r["rr_bpm"] for r in ok]
    disagree = sum(1 for r in ok if abs(r["bpm"] - r["rr_bpm"]) > BPM_DISAGREE)
    return {
        "window": label,
        "rows": n,
        "ok": len(ok),
        "zero": sum(1 for r in rows if r["cls"] == "zero"),
        "no_rr": sum(1 for r in rows if r["cls"] == "no_rr"),
        "pct_bad": round(100 * bad / n, 1) if n else None,
        "dropout_runs": len(run_list),
        "longest_dropout_s": max((b - a + 1 for a, b in run_list), default=0),
        "bpm_disagree_rows": disagree,
        "strap_mean": round(sum(strap) / len(strap), 1) if strap else None,
        "strap_max": max(strap) if strap else None,
        "rr_mean": round(sum(rr) / len(rr), 1) if rr else None,
        "rr_max": round(max(rr)) if rr else None,
    }


def trial_rows(trial: Path) -> list[dict]:
    rows = []
    for path in sorted(trial.glob("heart_rate/*hr_full_session*.csv")):
        if path.stem.endswith("_rr_clean"):
            continue
        with open(path, newline="", encoding="utf-8") as handle:
            for raw in csv.DictReader(handle):
                rows.append({"raw": raw,
                    "timestamp": float(raw["timestamp"]),
                    "phase": raw.get("phase", ""),
                    "bpm": int(float(raw["bpm"] or 0)),
                    "rr_intervals_ms": raw.get("rr_intervals_ms", ""),
                    "sensor_contact": raw.get("sensor_contact", ""),
                })
    rows.sort(key=lambda r: r["timestamp"])
    rr_filter = RRFilter()
    for r in rows:
        cls, rr_bpm = classify(r.pop("raw"), rr_filter)
        r["cls"], r["rr_bpm"] = cls, round(rr_bpm, 1) if rr_bpm else None
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--out", type=Path, default=None, help="directory for the summary CSVs")
    args = parser.parse_args()
    out_dir = args.out or args.data_root.parent / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)

    sessions, dropouts = [], []
    for trial in sorted(p for p in args.data_root.iterdir() if p.is_dir() and p.name.startswith("P")):
        manifest = next(trial.glob("*sync_manifest.json"), None)
        rows = trial_rows(trial)
        if not rows or manifest is None:
            continue
        pid = trial.name.split("_")[0]
        window = performance_window(json.loads(manifest.read_text(encoding="utf-8"))["events"])

        clean = trial / "heart_rate" / f"{pid}_hr_rr_clean.csv"
        with open(clean, "w", newline="", encoding="utf-8") as handle:
            fields = ["timestamp", "phase", "in_performance", "bpm", "rr_bpm", "cls", "rr_intervals_ms", "sensor_contact"]
            writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            for r in rows:
                r["in_performance"] = bool(window and window[0] <= r["timestamp"] < window[1])
                writer.writerow(r)

        windows = [("whole_session", rows)]
        if window:
            s, e = window
            windows += [
                ("performance", [r for r in rows if s <= r["timestamp"] < e]),
                (f"pre_{PRE_WINDOW_SEC:.0f}s", [r for r in rows if s - PRE_WINDOW_SEC <= r["timestamp"] < s]),
                (f"performance_plus_{POST_WINDOW_SEC:.0f}s", [r for r in rows if s <= r["timestamp"] < e + POST_WINDOW_SEC]),
            ]
        for label, subset in windows:
            if subset:
                sessions.append({"trial": trial.name, "pid": pid, **summarize(label, subset)})

        for a, b in runs(rows):
            start, end = rows[a]["timestamp"], rows[b]["timestamp"]
            overlap = bool(window and start < window[1] and end >= window[0])
            dropouts.append({
                "trial": trial.name, "pid": pid, "phase": rows[a]["phase"],
                "start_rel_perf_s": round(start - window[0], 1) if window else None,
                "duration_s": round(end - start + 1, 1),
                "kind": "zero" if all(r["cls"] == "zero" for r in rows[a:b + 1]) else
                        "frozen" if all(r["cls"] == "no_rr" for r in rows[a:b + 1]) else "mixed",
                "touches_performance": overlap,
                "held_bpm": sorted({r["bpm"] for r in rows[a:b + 1] if r["bpm"] > 0})[:5],
            })

    for name, data in (("hr_rr_qc_sessions.csv", sessions), ("hr_rr_qc_dropouts.csv", dropouts)):
        with open(out_dir / name, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(data[0]))
            writer.writeheader()
            writer.writerows(data)
        print(f"Wrote {out_dir / name} ({len(data)} rows)")

    print("\nDropouts touching the performance window:")
    for d in dropouts:
        if d["touches_performance"]:
            print(f"  {d['pid']:5} {d['kind']:6} {d['duration_s']:6.0f}s starting {d['start_rel_perf_s']:+7.1f}s from perf start"
                  f"  held bpm {d['held_bpm']}")
    print("\nPerformance windows with >10% bad rows or strap/RR disagreement:")
    for s in sessions:
        if s["window"] == "performance" and (s["pct_bad"] > 10 or s["bpm_disagree_rows"] > 5):
            print(f"  {s['pid']:5} bad {s['pct_bad']:5.1f}%  longest {s['longest_dropout_s']:3d}s  disagree {s['bpm_disagree_rows']:3d}"
                  f"  strap mean/max {s['strap_mean']}/{s['strap_max']}  rr mean/max {s['rr_mean']}/{s['rr_max']}")


if __name__ == "__main__":
    main()
