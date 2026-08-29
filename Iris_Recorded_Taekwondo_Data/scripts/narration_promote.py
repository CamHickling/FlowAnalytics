"""
Promote narration_units.edited.csv straight to narration_units.FROZEN.csv, then build the
coder template from it. Unlike narration_freeze.py, this does NOT renumber existing unit_ids
- it keeps every id you already have and only fills in ids that are blank or duplicated
(e.g. the rows you added when splitting a run-on). Use this when you want the frozen file to
mirror your edited file as literally as possible.

    python narration_promote.py                       # uses <root>/narration_units.edited.csv
    python narration_promote.py --in myfile.csv

Reads any Excel encoding (UTF-8 or Windows-1252). Writes UTF-8.
"""
import os, argparse
import pandas as pd

PER_UNIT_DIMENSIONS = ['perspective', 'sentiment', 'immersion']
SESSION_FACETS = ['facet_D1_action_awareness', 'facet_D2_concentration',
                  'facet_D3_loss_self_consciousness', 'facet_D4_sense_of_control',
                  'facet_D5_autotelic', 'facet_D6_time_transformation']

def _default_root():
    try:
        here = os.path.dirname(os.path.abspath(__file__))
        return os.path.dirname(here) if os.path.basename(here).lower() == 'scripts' else here
    except NameError:
        return os.getcwd()

def read_csv_smart(path, **kw):
    for enc in ('utf-8-sig', 'cp1252', 'latin-1'):
        try:
            return pd.read_csv(path, encoding=enc, **kw)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(path, encoding='utf-8', errors='replace', **kw)

def main():
    root = _default_root()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--in', dest='infile', default=None)
    ap.add_argument('--frozen', default=None)
    ap.add_argument('--template', default=None)
    args = ap.parse_args()

    infile = args.infile or os.path.join(root, 'narration_units.edited.csv')
    frozen_path = args.frozen or os.path.join(root, 'narration_units.FROZEN.csv')
    template_path = args.template or os.path.join(root, 'coder_TEMPLATE.csv')

    if not os.path.exists(infile):
        print(f"Missing {infile}"); return
    df = read_csv_smart(infile, dtype=str).fillna('')
    if not {'participant_id', 'unit_text'}.issubset(df.columns):
        print("Need at least participant_id + unit_text columns."); return

    df = df[df['unit_text'].str.strip() != ''].copy()
    if 'unit_id' not in df.columns:
        df['unit_id'] = ''
    df['unit_id'] = df['unit_id'].astype(str).str.strip()
    df['unit_text'] = df['unit_text'].str.strip()

    # only assign ids that are blank OR duplicated; keep all valid unique ids as-is
    n_fixed = 0
    for pid, g in df.groupby('participant_id', sort=False):
        used = set()
        # first pass: register the ids you already have that are unique & non-blank
        counts = g['unit_id'].value_counts()
        for idx, row in g.iterrows():
            uid = row['unit_id']
            keep = uid and uid != 'nan' and counts.get(uid, 0) == 1 and uid not in used
            if keep:
                used.add(uid)
        # second pass: fill blanks/dupes with the next free {pid}_NNN
        nxt = 1
        for idx in g.index:
            uid = df.at[idx, 'unit_id']
            if uid and uid != 'nan' and counts.get(uid, 0) == 1 and uid in used and df.at[idx, 'unit_id'] == uid:
                # already-valid id: mark consumed and continue
                continue
            while f"{pid}_{nxt:03d}" in used:
                nxt += 1
            new = f"{pid}_{nxt:03d}"
            df.at[idx, 'unit_id'] = new
            used.add(new); nxt += 1; n_fixed += 1

    if 'needs_review' not in df.columns:
        df['needs_review'] = False
    cols = ['participant_id', 'unit_id', 'unit_text', 'needs_review']
    df[cols].to_csv(frozen_path, index=False, encoding='utf-8')

    # template from the (now id-complete) frozen units
    rows = []
    for _, r in df.iterrows():
        for dim in PER_UNIT_DIMENSIONS:
            rows.append(dict(participant_id=r['participant_id'], unit_id=r['unit_id'],
                             unit_text=r['unit_text'], coder_id='', dimension=dim, label=''))
    for pid in df['participant_id'].unique():
        for facet in SESSION_FACETS:
            rows.append(dict(participant_id=pid, unit_id=f"{pid}_SESSION",
                             unit_text='', coder_id='', dimension=facet, label=''))
    pd.DataFrame(rows).to_csv(template_path, index=False, encoding='utf-8')

    still = int(df['needs_review'].astype(str).str.lower().isin(['true','1','yes']).sum())
    print(f"Promoted {len(df)} units across {df['participant_id'].nunique()} participants "
          f"({n_fixed} unit_id(s) were blank/duplicate and got filled).")
    print(f"  {frozen_path}")
    print(f"  {template_path}   (copy to coder_ANGELA.csv etc.)")
    if still:
        print(f"  [!] {still} row(s) still needs_review=True.")

if __name__ == '__main__':
    main()