"""
Segment narration transcripts into canonical thought-units (rubric section 1).

Author: AC Mejia
Version: 1.1
Date: 21-Aug-2026

This MUST run before both human coding and LLM labeling, because everyone has to code
the SAME units or their unit_ids won't align and kappa can't be computed. Segmentation is
deterministic (rule-based), so it is reproducible and identical across coders.

Reads:  <review folder>/P01_audio_narration.txt ... P50_audio_narration.txt
Writes: narration_units.csv          (participant_id, unit_id, unit_text)  <- the shared spine
        coder_TEMPLATE.csv           (blank; one row per unit x per-unit dimension +
                                      one row per session x facet; humans fill the 'label' column)

Segmentation rule: split on sentence-ending punctuation (. ? !) and on newlines; merge
fragments shorter than MIN_CHARS into the neighbour (filler / false starts); drop empties.
Whisper output is often lightly punctuated - eyeball narration_units.csv on the iteration
sample and hand-fix any run-ons before freezing (rubric section 7, calibration step).

Usage:
    python narration_make_units.py --review "path/to/review"
    python narration_make_units.py --review ../review --only P05 P41 P19   # iteration sample
"""

import os
import re
import sys
import glob
import argparse
import hashlib
import pandas as pd

MIN_CHARS = 12       # fragments shorter than this merge into the previous unit
LONG_WORDS = 28      # units longer than this are flagged needs_review (likely >1 idea)
SOFT_MIN_WORDS = 6   # only split at "and then" if both sides have at least this many words

import re as _re

def read_csv_smart(path, **kw):
    """Read a CSV trying UTF-8 (incl. BOM) then Windows-1252, so files saved from Excel
    as either encoding load without a UnicodeDecodeError (0x85 ellipsis, smart quotes...)."""
    import pandas as _pd
    for enc in ('utf-8-sig', 'cp1252', 'latin-1'):
        try:
            return _pd.read_csv(path, encoding=enc, **kw)
        except UnicodeDecodeError:
            continue
    return _pd.read_csv(path, encoding='utf-8', errors='replace', **kw)

_STRONG = _re.compile(r'(?<=[.?!])\s+|\.\.\.+|\s*\n+\s*')   # sentence enders, ellipses, newlines
_SOFT = _re.compile(r'\s+(?=and then\b)', _re.I)
_DISFLUENCY = _re.compile(r'\b(um+|uh+|erm)\b', _re.I)

def _clean(u):
    u = _DISFLUENCY.sub('', u)               # drop bare um/uh only; keep everything else verbatim
    u = _re.sub(r'\s+([,.])', r'\1', u)
    u = _re.sub(r'([,.])\1+', r'\1', u)
    u = _re.sub(r'\s{2,}', ' ', u).strip(' ,')
    return u

def _soft_split(seg):
    """Split at 'and then' only when both sides are substantial (never fragments)."""
    out, last = [], 0
    for m in _SOFT.finditer(seg):
        if len(seg[last:m.start()].split()) >= SOFT_MIN_WORDS and len(seg[m.start():].split()) >= SOFT_MIN_WORDS:
            out.append(seg[last:m.start()]); last = m.start()
    out.append(seg[last:])
    return out


def _default_root():
    try:
        here = os.path.dirname(os.path.abspath(__file__))
        return os.path.dirname(here) if os.path.basename(here).lower() == 'scripts' else here
    except NameError:
        return os.getcwd()

PER_UNIT_DIMENSIONS = ['perspective', 'sentiment', 'immersion']
SESSION_FACETS = [
    'facet_D1_action_awareness',
    'facet_D2_concentration',
    'facet_D3_loss_self_consciousness',
    'facet_D4_sense_of_control',
    'facet_D5_autotelic',
    'facet_D6_time_transformation',
]


def segment(text):
    """Split into thought-units on STRONG boundaries only, so run-ons are never
    fragmented. Returns list of (unit_text, needs_review) where needs_review flags
    units still long enough to likely contain >1 idea (unpunctuated ASR) -> a human
    splits those during calibration. Text is kept verbatim except bare um/uh disfluencies."""
    text = text.replace('\r', '\n')
    units = []
    for seg in _STRONG.split(text):
        if not seg or not seg.strip():
            continue
        for piece in _soft_split(seg.strip()):
            u = _clean(piece)
            if not u:
                continue
            if len(u) < MIN_CHARS and units:
                units[-1] = (units[-1] + ' ' + u).strip()
            else:
                units.append(u)
    return [(u, len(u.split()) > LONG_WORDS) for u in units]


def participant_from_name(fname):
    m = re.match(r'(P\d+[A-Za-z]*)', os.path.basename(fname))
    return m.group(1) if m else os.path.splitext(os.path.basename(fname))[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--review', default=None,
                    help='folder to search. If omitted, auto-detects the data root and '
                         'searches it RECURSIVELY (transcripts live in per-session review/ '
                         'subfolders, like the HR files).')
    ap.add_argument('--only', nargs='*', help='restrict to these participant ids (iteration sample)')
    ap.add_argument('--pattern', default='*_audio_narration.txt')
    ap.add_argument('--outdir', default=None,
                    help='where to write the CSVs (default: the data root being searched)')
    ap.add_argument('--flat', action='store_true',
                    help='search only the given --review folder, non-recursively')
    ap.add_argument('--prefer', default=None,
                    help="when a participant has copies in two folders, keep the one whose path "
                         "contains this string (e.g. 'review' or 'transcripts'). Default: warn "
                         "and compare contents so you can decide.")
    args = ap.parse_args()

    root = args.review or _default_root()
    if args.flat:
        files = sorted(glob.glob(os.path.join(root, args.pattern)))
    else:
        # recurse: matches ...\P01_...\review\P01_audio_narration.txt at any depth
        files = sorted(glob.glob(os.path.join(root, '**', args.pattern), recursive=True))
    print(f"Searching {'(flat) ' if args.flat else '(recursive) '}{root}")
    EXCLUDE = {'P02B', 'P10B', 'P18B', 'P27B'}   # retries/freestyle, per the cardiac exclusions
    files = [f for f in files if participant_from_name(f).upper() not in EXCLUDE]
    if args.only:
        keep = {p.upper() for p in args.only}
        files = [f for f in files if participant_from_name(f).upper() in keep]
    # one transcript per participant. Duplicates are common here (a per-session review/ copy
    # AND a central transcripts/ copy). Compare CONTENTS and report, rather than picking blindly.
    def _digest(path):
        with open(path, 'rb') as fh:
            return hashlib.md5(fh.read()).hexdigest()

    by_pid = {}
    for f in files:
        by_pid.setdefault(participant_from_name(f).upper(), []).append(f)

    deduped, n_identical, n_conflict = [], 0, 0
    for pid, paths in sorted(by_pid.items()):
        if len(paths) == 1:
            deduped.append(paths[0]); continue
        # choose by --prefer if given, else first sorted
        chosen = paths[0]
        if args.prefer:
            pref = [p for p in paths if args.prefer.lower() in p.lower()]
            if pref:
                chosen = pref[0]
        digs = {_digest(p) for p in paths}
        if len(digs) == 1:
            n_identical += 1
            print(f"  [dup-ok]  {pid}: {len(paths)} identical copies -> using "
                  f"{os.path.relpath(chosen, root)}")
        else:
            n_conflict += 1
            print(f"  [CONFLICT] {pid}: {len(paths)} copies DIFFER in content:")
            for p in paths:
                tag = ' <- USING' if p == chosen else ''
                print(f"             {os.path.relpath(p, root)}  (md5 {_digest(p)[:8]}){tag}")
        deduped.append(chosen)
    files = deduped
    if n_identical or n_conflict:
        print(f"\n  Duplicate summary: {n_identical} participant(s) had identical copies, "
              f"{n_conflict} had CONFLICTING copies.")
        if n_conflict:
            print("  >>> Resolve conflicts before freezing: decide which folder is authoritative")
            print("  >>> and re-run with --prefer review   (or --prefer transcripts).")
    if not files:
        print(f"No transcripts matching {args.pattern} in {root}")
        if args.only:
            print(f"  (filtered to {args.only})")
        return

    unit_rows, template_rows = [], []
    for f in files:
        pid = participant_from_name(f)
        with open(f, encoding='utf-8', errors='replace') as fh:
            text = fh.read()
        units = segment(text)
        n_flag = 0
        for i, (u, needs_review) in enumerate(units, 1):
            uid = f"{pid}_{i:03d}"
            n_flag += int(needs_review)
            unit_rows.append(dict(participant_id=pid, unit_id=uid, unit_text=u,
                                  needs_review=needs_review))
            for dim in PER_UNIT_DIMENSIONS:
                template_rows.append(dict(participant_id=pid, unit_id=uid, unit_text=u,
                                          needs_review=needs_review, coder_id='', dimension=dim, label=''))
        # one session-level row per facet
        for facet in SESSION_FACETS:
            template_rows.append(dict(participant_id=pid, unit_id=f"{pid}_SESSION",
                                      unit_text='', coder_id='', dimension=facet, label=''))
        print(f"  {pid}: {len(units)} units" + (f"  ({n_flag} flagged for manual split)" if n_flag else ""))

    outdir = args.outdir or root
    os.makedirs(outdir, exist_ok=True)
    units_path = os.path.join(outdir, 'narration_units.csv')
    template_path = os.path.join(outdir, 'coder_TEMPLATE.csv')
    pd.DataFrame(unit_rows).to_csv(units_path, index=False)
    pd.DataFrame(template_rows).to_csv(template_path, index=False)
    n_review = sum(1 for r in unit_rows if r.get('needs_review'))
    print(f"\n{len(files)} transcripts -> {len(unit_rows)} units  ({n_review} flagged needs_review "
          f"= {100*n_review/max(len(unit_rows),1):.0f}% -> split these by hand in calibration, "
          f"or ask for the LLM-assisted splitter if there are too many)")
    print(f"  {units_path}   (shared spine for LLM + humans)")
    print(f"  {template_path}   (duplicate per coder -> coder_ANGELA.csv etc.; fill 'label')")


if __name__ == '__main__':
    main()