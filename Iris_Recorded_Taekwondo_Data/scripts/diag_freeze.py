"""Diagnose why FROZEN doesn't match your edits. Run:
   python diag_freeze.py --edited "PATH\to\narration_units.edited.csv"
"""
import argparse, os, pandas as pd
ap=argparse.ArgumentParser(); ap.add_argument('--edited',required=True)
ap.add_argument('--frozen',default=None); a=ap.parse_args()
root=os.path.dirname(os.path.abspath(a.edited))
frozen=a.frozen or os.path.join(root,'narration_units.FROZEN.csv')

print("EDITED file:", a.edited)
print("  exists:", os.path.exists(a.edited),
      "| modified:", pd.Timestamp(os.path.getmtime(a.edited),unit='s') if os.path.exists(a.edited) else "-")
e=pd.read_csv(a.edited,dtype=str).fillna('')
print("  columns:", list(e.columns))
print("  rows:", len(e), "| non-empty unit_text:", (e['unit_text'].str.strip()!='').sum() if 'unit_text' in e else "NO unit_text COLUMN!")
if 'unit_text' in e:
    print("  first 3 unit_text values you edited:")
    for t in e['unit_text'].head(3): print("     ", repr(t[:80]))

if os.path.exists(frozen):
    print("\nFROZEN file:", frozen)
    print("  modified:", pd.Timestamp(os.path.getmtime(frozen),unit='s'))
    f=pd.read_csv(frozen,dtype=str).fillna('')
    print("  first 3 unit_text values in frozen:")
    for t in f['unit_text'].head(3): print("     ", repr(t[:80]))
    # is frozen newer than edited? if not, freeze ran BEFORE your last save
    if os.path.getmtime(frozen) < os.path.getmtime(a.edited):
        print("\n  >>> FROZEN is OLDER than EDITED: you edited/saved AFTER the last freeze.")
        print("      Re-run freeze on this edited file.")
    else:
        et=set(e['unit_text'].str.strip()) if 'unit_text' in e else set()
        ft=set(f['unit_text'].str.strip())
        missing=[t for t in et if t and t not in ft]
        print(f"\n  edited lines NOT found in frozen: {len(missing)}")
        for t in missing[:5]: print("     MISSING:", repr(t[:80]))
        if not missing:
            print("  >>> All your edited text IS in the frozen file. The mismatch is likely ROW ORDER,")
            print("      not content - open both sorted by participant_id and compare.")
else:
    print("\nNo FROZEN file next to the edited one -> freeze wrote it somewhere else,")
    print("or was run with a different --frozen path / working directory.")