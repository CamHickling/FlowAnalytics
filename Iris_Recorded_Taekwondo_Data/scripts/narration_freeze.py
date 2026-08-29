"""
Freeze the segmentation, and (re)build the coder template from the FROZEN units.

The workflow this supports:

  1. make_units.py  -> narration_units.csv         (auto, overwritable DRAFT)
  2. you hand-fix the needs_review run-ons in a COPY (see --help below)
  3. narration_freeze.py  -> narration_units.FROZEN.csv + coder_TEMPLATE.csv
                           (renumbers unit_ids, regenerates the template to match)
  4. everyone codes the FROZEN units; make_units.py can be re-run without touching them

Why a separate frozen file: narration_units.csv is an OUTPUT of make_units.py and is
overwritten every run. If you edit it in place, the next run destroys your splits. The
frozen file is an INPUT nothing regenerates, so it is safe to code against.

------------------------------------------------------------------------------
HOW TO ADDRESS needs_review (the efficient way):
  - Open narration_units.csv in Excel. Filter needs_review == True.
  - To SPLIT a long unit into two: duplicate its row, and edit unit_text in each so
    each row holds one idea. Leave unit_id blank in BOTH (freeze renumbers).
  - To MERGE two over-split units: delete one row, paste its text onto the other.
  - You do NOT need to renumber anything or keep unit_ids consistent - freeze does it.
  - Set needs_review to False (or blank) on rows you've resolved.
  Save As  narration_units.edited.csv  (keep the original as a backup).

Then:  python narration_freeze.py --in narration_units.edited.csv
------------------------------------------------------------------------------
"""

import os
import argparse
import pandas as pd

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


PER_UNIT_DIMENSIONS = ['perspective', 'sentiment', 'immersion']
SESSION_FACETS = [
    'facet_D1_action_awareness', 'facet_D2_concentration',
    'facet_D3_loss_self_consciousness', 'facet_D4_sense_of_control',
    'facet_D5_autotelic', 'facet_D6_time_transformation',
]


def _default_root():
    try:
        here = os.path.dirname(os.path.abspath(__file__))
        return os.path.dirname(here) if os.path.basename(here).lower() == 'scripts' else here
    except NameError:
        return os.getcwd()


def main():
    root = _default_root()
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--in', dest='infile', default=None,
                    help='units CSV to freeze (default: <root>/narration_units.csv)')
    ap.add_argument('--frozen', default=None,
                    help='output frozen units (default: <root>/narration_units.FROZEN.csv)')
    ap.add_argument('--template', default=None,
                    help='output coder template (default: <root>/coder_TEMPLATE.csv)')
    args = ap.parse_args()

    infile = args.infile or os.path.join(root, 'narration_units.csv')
    frozen_path = args.frozen or os.path.join(root, 'narration_units.FROZEN.csv')
    template_path = args.template or os.path.join(root, 'coder_TEMPLATE.csv')

    if not os.path.exists(infile):
        print(f"Missing {infile}. Run narration_make_units.py first.")
        return
    df = read_csv_smart(infile, dtype=str).fillna('')
    if not {'participant_id', 'unit_text'}.issubset(df.columns):
        print(f"{infile} needs at least participant_id + unit_text columns.")
        return

    # drop blank-text rows (e.g. a row someone emptied while merging)
    df = df[df['unit_text'].str.strip() != ''].copy()

    # RENUMBER unit_ids per participant, in file order -> stable, gap-free, matches edits
    frozen_units, template_rows = [], []
    still_flagged = 0
    for pid, g in df.groupby('participant_id', sort=False):
        for i, (_, row) in enumerate(g.iterrows(), 1):
            uid = f"{pid}_{i:03d}"
            nr = str(row.get('needs_review', '')).strip().lower() in ('true', '1', 'yes')
            still_flagged += int(nr)
            frozen_units.append(dict(participant_id=pid, unit_id=uid,
                                     unit_text=row['unit_text'].strip(), needs_review=nr))
            for dim in PER_UNIT_DIMENSIONS:
                template_rows.append(dict(participant_id=pid, unit_id=uid,
                                          unit_text=row['unit_text'].strip(),
                                          coder_id='', dimension=dim, label=''))
        for facet in SESSION_FACETS:
            template_rows.append(dict(participant_id=pid, unit_id=f"{pid}_SESSION",
                                      unit_text='', coder_id='', dimension=facet, label=''))

    fu = pd.DataFrame(frozen_units)
    fu.to_csv(frozen_path, index=False)
    pd.DataFrame(template_rows).to_csv(template_path, index=False)

    print(f"Froze {len(fu)} units across {fu.participant_id.nunique()} participants.")
    print(f"  {frozen_path}   <- the coding spine (point the LLM labeler + humans here)")
    print(f"  {template_path}   <- regenerated to match; copy to coder_ANGELA.csv etc.")
    if still_flagged:
        print(f"\n  [!] {still_flagged} unit(s) still marked needs_review. Split/resolve them and "
              f"re-freeze, OR accept them as-is if they're genuinely one idea.")
    else:
        print("\n  All needs_review units resolved. Safe to code.")


if __name__ == '__main__':
    main()