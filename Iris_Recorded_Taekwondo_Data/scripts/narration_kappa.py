"""
Inter-rater reliability for the narration coding (Phase 4 of the rubric protocol).

Reads every coder file matching coder_*.csv plus llm_labels.csv (all in the long-format
schema from rubric §9) and computes, PER DIMENSION:

  - Cohen's kappa            (2 coders; nominal)
  - Cohen's weighted kappa   (2 coders; ORDINAL dims e.g. sentiment - linear & quadratic)
  - Fleiss' kappa            (3+ coders; nominal)
  - raw % agreement
  - PABAK                    (prevalence-adjusted; for the rare-label dims C/D - see rubric §6)

and reports human<->human separately from human<->LLM, because they answer different
questions (is the RUBRIC reliable vs can the LLM be trusted as a coder).

All kappa variants are implemented from scratch in numpy - no sklearn/statsmodels needed,
so it runs in the minimal lab env. A --selftest validates them against textbook values.

Usage:
    python narration_kappa.py              # scans ./ for coder_*.csv and llm_labels.csv
    python narration_kappa.py --selftest   # verify the kappa math
"""

import os
import sys
import glob
import itertools
import numpy as np
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



def _default_root():
    try:
        here = os.path.dirname(os.path.abspath(__file__))
        return os.path.dirname(here) if os.path.basename(here).lower() == 'scripts' else here
    except NameError:
        return os.getcwd()

# ordinal dimensions -> use weighted kappa; everything else is nominal
ORDINAL_DIMS = {'sentiment'}
LLM_CODER_ID = 'LLM'
LANDIS_KOCH = [(-1, 0.0, 'poor'), (0.0, 0.20, 'slight'), (0.20, 0.40, 'fair'),
               (0.40, 0.60, 'moderate'), (0.60, 0.80, 'substantial'), (0.80, 1.01, 'almost perfect')]


def _band(k):
    if not np.isfinite(k):
        return 'undefined'
    for lo, hi, name in LANDIS_KOCH:
        if lo < k <= hi or (lo == -1 and k <= hi):
            return name
    return '?'


# --------------------------------------------------------------------------- #
#  KAPPA IMPLEMENTATIONS (numpy only)
# --------------------------------------------------------------------------- #
def confusion(a, b, cats):
    idx = {c: i for i, c in enumerate(cats)}
    m = np.zeros((len(cats), len(cats)))
    for x, y in zip(a, b):
        m[idx[x], idx[y]] += 1
    return m


def cohen_kappa(a, b, cats, weights=None):
    """Cohen's kappa for two coders. weights in {None,'linear','quadratic'} (ordinal)."""
    n = len(a)
    if n == 0:
        return np.nan
    O = confusion(a, b, cats) / n
    r = O.sum(axis=1, keepdims=True)
    c = O.sum(axis=0, keepdims=True)
    E = r @ c
    k = len(cats)
    if weights is None:
        W = 1 - np.eye(k)                    # disagreement weight = 1 off-diagonal
    else:
        i = np.arange(k).reshape(-1, 1)
        j = np.arange(k).reshape(1, -1)
        d = np.abs(i - j).astype(float)
        W = d if weights == 'linear' else d ** 2
        if W.max() > 0:
            W = W / W.max()
    denom = (W * E).sum()
    if denom == 0:
        return np.nan
    return 1 - (W * O).sum() / denom


def fleiss_kappa(counts):
    """Fleiss' kappa. counts: (n_units x n_categories), row = how many raters chose each
    category for that unit (rows must sum to a constant n_raters)."""
    counts = np.asarray(counts, float)
    N, k = counts.shape
    n = counts.sum(axis=1)
    if not np.allclose(n, n[0]) or n[0] < 2:
        return np.nan
    n = n[0]
    P_i = (np.sum(counts ** 2, axis=1) - n) / (n * (n - 1))
    P_bar = P_i.mean()
    p_j = counts.sum(axis=0) / (N * n)
    P_e = np.sum(p_j ** 2)
    if np.isclose(P_e, 1.0):
        return np.nan
    return (P_bar - P_e) / (1 - P_e)


def raw_agreement(a, b):
    return np.mean([x == y for x, y in zip(a, b)]) if len(a) else np.nan


def pabak(a, b, cats):
    """Prevalence-adjusted bias-adjusted kappa (Byrt 1993). For rare-label dims where
    Cohen's kappa is deflated by high chance agreement. Generalized to k categories:
    PABAK = (k*p_o - 1)/(k - 1)."""
    p_o = raw_agreement(a, b)
    k = len(cats)
    if k < 2:
        return np.nan
    return (k * p_o - 1) / (k - 1)


# --------------------------------------------------------------------------- #
#  DATA
# --------------------------------------------------------------------------- #
def load_coders(root):
    files = sorted(glob.glob(os.path.join(root, 'coder_*.csv')))
    llm = os.path.join(root, 'llm_labels.csv')
    if os.path.exists(llm):
        files.append(llm)
    frames = []
    for f in files:
        d = read_csv_smart(f, dtype=str).fillna('')
        need = {'unit_id', 'coder_id', 'dimension', 'label'}
        if not need.issubset(d.columns):
            print(f"  [skip] {os.path.basename(f)} missing columns {need - set(d.columns)}")
            continue
        frames.append(d[['unit_id', 'coder_id', 'dimension', 'label']])
    if not frames:
        return None
    return pd.concat(frames, ignore_index=True)


def _aligned(df, dim, coders):
    """Return the label matrix for `dim` on units coded by ALL `coders` (listwise)."""
    sub = df[(df.dimension == dim) & (df.coder_id.isin(coders))]
    wide = sub.pivot_table(index='unit_id', columns='coder_id', values='label',
                           aggfunc='first')
    wide = wide.dropna(subset=coders)
    return wide


def _fleiss_table(wide, cats):
    idx = {c: i for i, c in enumerate(cats)}
    tab = np.zeros((len(wide), len(cats)))
    for r, (_, row) in enumerate(wide.iterrows()):
        for v in row.values:
            tab[r, idx[v]] += 1
    return tab


def analyse(df):
    dims = sorted(df.dimension.unique())
    all_coders = sorted(df.coder_id.unique())
    humans = [c for c in all_coders if c != LLM_CODER_ID]
    has_llm = LLM_CODER_ID in all_coders
    rows = []

    for dim in dims:
        cats = sorted(df.loc[df.dimension == dim, 'label'].unique())
        ordinal = dim in ORDINAL_DIMS
        # try to order ordinal categories numerically for weighting
        if ordinal:
            try:
                cats = sorted(cats, key=lambda x: float(x))
            except ValueError:
                ordinal = False

        def pair(c1, c2, kind):
            wide = _aligned(df, dim, [c1, c2])
            if len(wide) < 2:
                return None
            a, b = wide[c1].tolist(), wide[c2].tolist()
            rec = dict(dimension=dim, comparison=f"{c1} vs {c2}", kind=kind,
                       n_units=len(wide), n_categories=len(cats),
                       raw_agreement=round(raw_agreement(a, b), 3),
                       cohen_kappa=round(cohen_kappa(a, b, cats), 3))
            if ordinal:
                rec['kappa_linear'] = round(cohen_kappa(a, b, cats, 'linear'), 3)
                rec['kappa_quadratic'] = round(cohen_kappa(a, b, cats, 'quadratic'), 3)
            rec['pabak'] = round(pabak(a, b, cats), 3)
            rec['band'] = _band(rec['cohen_kappa'])
            return rec

        # human <-> human (all pairs)
        for c1, c2 in itertools.combinations(humans, 2):
            r = pair(c1, c2, 'human-human')
            if r:
                rows.append(r)
        # Fleiss across all humans if 3+
        if len(humans) >= 3:
            wide = _aligned(df, dim, humans)
            if len(wide) >= 2:
                fk = fleiss_kappa(_fleiss_table(wide, cats))
                rows.append(dict(dimension=dim, comparison=f"Fleiss ({len(humans)} humans)",
                                 kind='human-human', n_units=len(wide), n_categories=len(cats),
                                 raw_agreement=np.nan, cohen_kappa=round(fk, 3),
                                 pabak=np.nan, band=_band(fk)))
        # human <-> LLM (each human)
        if has_llm:
            for c in humans:
                r = pair(c, LLM_CODER_ID, 'human-LLM')
                if r:
                    rows.append(r)

    return pd.DataFrame(rows)


def report(res):
    if res is None or res.empty:
        print("No comparable data.")
        return
    pd.set_option('display.width', 200, 'display.max_columns', 30)
    for kind in ['human-human', 'human-LLM']:
        block = res[res.kind == kind]
        if block.empty:
            continue
        print(f"\n{'='*78}\n{kind.upper()}\n{'='*78}")
        cols = [c for c in ['dimension', 'comparison', 'n_units', 'n_categories',
                            'raw_agreement', 'cohen_kappa', 'kappa_linear', 'kappa_quadratic',
                            'pabak', 'band'] if c in block.columns]
        print(block[cols].to_string(index=False))

    print(f"\n{'='*78}\nREADOUT\n{'='*78}")
    hh = res[res.kind == 'human-human']
    for dim in sorted(res.dimension.unique()):
        d = hh[hh.dimension == dim]
        if d.empty:
            continue
        k = d.cohen_kappa.mean()
        note = ''
        if dim in ('immersion',) or dim.startswith('facet'):
            gap = d.pabak.mean() - k if d.pabak.notna().any() else np.nan
            if np.isfinite(gap) and gap > 0.2:
                note = f"  (rare-label: raw agr {d.raw_agreement.mean():.2f}, PABAK {d.pabak.mean():.2f} >> kappa -> prevalence-deflated, see rubric section 6)"
        print(f"  {dim:36s} human-human kappa = {k:+.2f} [{_band(k)}]{note}")


def selftest():
    print("SELF-TEST (textbook values)")
    # Cohen: classic 2x2 with a=b=10, off=5/5 over n=... use a known set
    a = list('YYYYNNNNNN'); b = list('YYNNNNNNNN')  # 8/10 agree; a:4Y6N b:2Y8N
    k = cohen_kappa(a, b, ['Y', 'N'])
    # Po=0.8, Pe=(.4*.2)+(.6*.8)=0.56 -> (0.8-0.56)/(1-0.56)=0.5455
    print(f"  Cohen kappa      = {k:.4f}   expect 0.5455   {'OK' if abs(k-0.5455)<1e-3 else 'FAIL'}")
    # Weighted (quadratic) sanity: perfect agreement -> 1
    aa = list('12321'); bb = list('12321')
    print(f"  Weighted (perf.) = {cohen_kappa(aa,bb,['1','2','3'],'quadratic'):.4f}   expect 1.0000")
    # Fleiss: Fleiss (1971) worked example fragment -> use a small known table
    # 4 raters, 3 units, 2 categories
    tab = np.array([[4, 0], [0, 4], [2, 2]], float)  # unit3 split
    fk = fleiss_kappa(tab)
    # P1=1,P2=1,P3=(4+4-4)/(4*3)=0.333; Pbar=0.7778; pj=(6/12,6/12)->Pe=0.5; (0.7778-0.5)/0.5=0.5556
    print(f"  Fleiss kappa     = {fk:.4f}   expect 0.5556   {'OK' if abs(fk-0.5556)<1e-3 else 'FAIL'}")
    # PABAK: 90% agreement, 2 cats -> 0.8
    aa = ['N']*90 + ['Y']*10; bb = ['N']*90 + ['Y']*10
    print(f"  PABAK (100% agr) = {pabak(aa,bb,['Y','N']):.4f}   expect 1.0000")


if __name__ == '__main__':
    if '--selftest' in sys.argv:
        selftest()
    else:
        root = _default_root()
        print(f'Scanning {root} for coder_*.csv and llm_labels.csv')
        df = load_coders(root)
        if df is None:
            print("No coder_*.csv or llm_labels.csv found in", root)
            print("Expected long-format files (rubric section 9): unit_id, coder_id, dimension, label")
        else:
            print(f"Loaded {len(df)} label rows from "
                  f"{df.coder_id.nunique()} coders: {sorted(df.coder_id.unique())}")
            res = analyse(df)
            report(res)
            out = os.path.join(root, 'narration_reliability_results.csv')
            if res is not None and not res.empty:
                res.to_csv(out, index=False)
                print(f"\n-> {out}")