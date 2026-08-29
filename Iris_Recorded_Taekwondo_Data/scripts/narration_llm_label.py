"""
LLM labeler (Phase 2 of the rubric protocol).

Applies the FROZEN rubric to the pre-segmented units in narration_units.csv and emits
llm_labels.csv in the same long-format schema humans use. The LLM is treated as one more
coder whose agreement with humans is then measured (narration_kappa.py) - it does NOT get
to invent categories or re-segment.

Design choices that matter:
  - Labels the SAME units the humans code (reads narration_units.csv) -> unit_ids align.
  - Codes one SESSION at a time (all its units in one call) so the model sees context, but
    is constrained to return exactly one label per unit per dimension from the codebook.
  - Output is strict JSON, validated against the codebook; anything invalid is set to a
    sentinel ('?') and counted, never silently guessed.
  - SDK v1.0 removed temperature/top_p/top_k, so runs are not forced to temp 0;
    outputs vary slightly run-to-run. Freeze the LLM labels once and reuse them.

Requires the anthropic SDK and ANTHROPIC_API_KEY in the environment:
    pip install anthropic
    setx ANTHROPIC_API_KEY "sk-..."     (Windows)  /  export ...  (mac/linux)

Usage:
    python narration_llm_label.py                       # labels every session in narration_units.csv
    python narration_llm_label.py --only P05 P41 P19    # iteration sample
    python narration_llm_label.py --rubric ../narration_coding_rubric.md
"""

import os
import re
import sys
import json
import time
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



def _default_root():
    try:
        here = os.path.dirname(os.path.abspath(__file__))
        return os.path.dirname(here) if os.path.basename(here).lower() == 'scripts' else here
    except NameError:
        return os.getcwd()

MODEL = "claude-sonnet-4-6"
CODER_ID = "LLM"

# The codebook the model MUST choose from. Keep in lock-step with the rubric.
CODEBOOK = {
    'perspective': ['F', 'O', 'M', 'N'],
    'sentiment':   ['-2', '-1', '0', '1', '2'],
    'immersion':   ['A', 'R', 'B', 'N'],
}
FACETS = {
    'facet_D1_action_awareness': ['P', 'C', 'N'],
    'facet_D2_concentration': ['P', 'C', 'N'],
    'facet_D3_loss_self_consciousness': ['P', 'C', 'N'],
    'facet_D4_sense_of_control': ['P', 'C', 'N'],
    'facet_D5_autotelic': ['P', 'C', 'N'],
    'facet_D6_time_transformation': ['P', 'C', 'N'],
}

SYSTEM = """You are a trained research coder applying a FIXED codebook to self-narration from a \
flow study. Athletes narrate while re-watching video of their own taekwondo poomsae performance.

You are ONE coder among several; your labels will be checked against human coders. Follow these \
rules exactly:

1. Use ONLY the label values given. Never invent categories. If genuinely uncodable, use "?".
2. Do not re-segment. Label each unit you are given, by its unit_id.
3. Per unit, label three dimensions:
   - perspective: F=field/first-person experiential ("I felt locked in"); O=observer/self as \
object seen from outside ("you can see my leg drift", "my body just moved"); M=both in one unit; \
N=no self-reference (logistics, questions).
   - sentiment (toward the PERFORMANCE, ordinal): 2=strongly positive, 1=mildly positive, \
0=neutral/factual, -1=mildly negative, -2=strongly negative. Flat technical corrections are 0.
   - immersion: A=reports being absorbed/automatic DURING THE PERFORMANCE ("I wasn't thinking"); \
R=reports deliberate thinking/planning DURING THE PERFORMANCE ("I was counting", "trying to \
remember the next move"); B=both; N=no evidence about the performance-time mental state.
   CRITICAL: immersion is about the mental state DURING THE ORIGINAL PERFORMANCE, never the \
state while narrating. "I can't remember what's next" as a narration-time struggle is N, not R.
4. Per SESSION, judge six flow facets from the whole narration: P=present evidence, \
C=contradicted (explicit counter-evidence), N=no evidence.
   D1 action-awareness merging/automaticity; D2 total concentration; D3 loss of \
self-consciousness (not caring how they looked / forgetting observers); D4 sense of control; \
D5 autotelic (enjoyed it for its own sake); D6 transformation of time (flew by / slow motion).

Return ONLY valid JSON, no prose, no markdown fences."""

USER_TEMPLATE = """Session: {pid}

Units to label:
{units_block}

Return JSON exactly in this shape:
{{
  "units": [
    {{"unit_id": "<id>", "perspective": "<F|O|M|N>", "sentiment": "<-2|-1|0|1|2>", "immersion": "<A|R|B|N>"}}
  ],
  "facets": {{
    "facet_D1_action_awareness": "<P|C|N>",
    "facet_D2_concentration": "<P|C|N>",
    "facet_D3_loss_self_consciousness": "<P|C|N>",
    "facet_D4_sense_of_control": "<P|C|N>",
    "facet_D5_autotelic": "<P|C|N>",
    "facet_D6_time_transformation": "<P|C|N>"
  }}
}}"""


def build_prompt(pid, units):
    block = "\n".join(f'{r.unit_id}: "{r.unit_text}"' for r in units.itertuples())
    return USER_TEMPLATE.format(pid=pid, units_block=block)


def parse_response(text, pid, units):
    """Parse + VALIDATE against the codebook. Returns (rows, n_invalid)."""
    # tolerate stray fences/prose despite instructions
    m = re.search(r'\{.*\}', text, re.DOTALL)
    if not m:
        raise ValueError("no JSON object in response")
    data = json.loads(m.group(0))
    rows, n_invalid = [], 0
    valid_ids = set(units.unit_id)

    seen = set()
    for u in data.get('units', []):
        uid = u.get('unit_id', '')
        if uid not in valid_ids:
            continue
        seen.add(uid)
        text_u = units.loc[units.unit_id == uid, 'unit_text'].iloc[0]
        for dim, allowed in CODEBOOK.items():
            lab = str(u.get(dim, '?')).strip()
            if lab not in allowed:
                lab = '?'
                n_invalid += 1
            rows.append(dict(participant_id=pid, unit_id=uid, unit_text=text_u,
                             coder_id=CODER_ID, dimension=dim, label=lab))
    # units the model forgot -> mark uncodable rather than dropping (keeps alignment honest)
    for uid in valid_ids - seen:
        text_u = units.loc[units.unit_id == uid, 'unit_text'].iloc[0]
        for dim in CODEBOOK:
            rows.append(dict(participant_id=pid, unit_id=uid, unit_text=text_u,
                             coder_id=CODER_ID, dimension=dim, label='?'))
            n_invalid += 1

    facets = data.get('facets', {})
    for facet, allowed in FACETS.items():
        lab = str(facets.get(facet, '?')).strip()
        if lab not in allowed:
            lab = '?'
            n_invalid += 1
        rows.append(dict(participant_id=pid, unit_id=f"{pid}_SESSION", unit_text='',
                         coder_id=CODER_ID, dimension=facet, label=lab))
    return rows, n_invalid


def call_model(client, pid, units, retries=3):
    prompt = build_prompt(pid, units)
    last = None
    for attempt in range(retries):
        try:
            resp = client.messages.create(
                model=MODEL, max_tokens=4000,
                system=SYSTEM, messages=[{"role": "user", "content": prompt}])
            text = "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")
            return parse_response(text, pid, units)
        except Exception as e:                       # JSON or API error -> back off and retry
            last = e
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"{pid}: failed after {retries} attempts ({last})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--units', default=None, help='default: <data root>/narration_units.csv')
    ap.add_argument('--rubric', default='narration_coding_rubric.md',
                    help='(informational; the codebook is embedded above)')
    ap.add_argument('--only', nargs='*')
    ap.add_argument('--out', default=None, help='default: <data root>/llm_labels.csv')
    args = ap.parse_args()

    root = _default_root()
    # prefer the frozen spine once it exists, so labeling always uses the same units humans code
    _frozen = os.path.join(root, 'narration_units.FROZEN.csv')
    units_path = args.units or (_frozen if os.path.exists(_frozen)
                                else os.path.join(root, 'narration_units.csv'))
    out_path = args.out or os.path.join(root, 'llm_labels.csv')
    if not os.path.exists(units_path):
        print(f"Missing {units_path}. Run narration_make_units.py first.")
        return
    print(f"Reading units: {units_path}")
    units_all = read_csv_smart(units_path, dtype=str).fillna('')
    if args.only:
        keep = {p.upper() for p in args.only}
        units_all = units_all[units_all.participant_id.str.upper().isin(keep)]
    if units_all.empty:
        print("No units to label after filtering.")
        return

    try:
        import anthropic
    except ImportError:
        print("The 'anthropic' package is not installed.  pip install anthropic")
        return
    if not os.environ.get('ANTHROPIC_API_KEY'):
        print("ANTHROPIC_API_KEY is not set in the environment.")
        return
    client = anthropic.Anthropic()

    all_rows, total_invalid = [], 0
    for pid, units in units_all.groupby('participant_id'):
        try:
            rows, n_invalid = call_model(client, pid, units.reset_index(drop=True))
            all_rows.extend(rows)
            total_invalid += n_invalid
            flag = f"  [{n_invalid} uncodable/invalid -> '?']" if n_invalid else ""
            print(f"  {pid}: {len(units)} units labeled{flag}")
        except Exception as e:
            print(f"  [ERROR] {e}")

    if all_rows:
        pd.DataFrame(all_rows).to_csv(out_path, index=False)
        print(f"\n-> {out_path}   ({total_invalid} cells set to '?' across the sample)")
        if total_invalid:
            print("  '?' cells are EXCLUDED pairwise in narration_kappa.py; a high count means "
                  "the rubric/prompt needs tightening before trusting the LLM labels.")


if __name__ == '__main__':
    main()