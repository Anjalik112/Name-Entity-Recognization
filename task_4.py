#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Step 4 — ADR-only evaluation with MedDRA gold

- Gold:   <original-dir>/<gold-subdir>/<file>.ann   (all T-lines treated as ADR)
- Pred:   <predicted-dir>/<file>.ann                (keep ADR-only, case-insensitive)
- Text:   <text-dir>/<file>.txt                     (for printing snippets)

Matching:
  --match strict   : exact (label, start, end)
  --match overlap  : TP if any character overlap (greedy, one pred ↔ one gold)

Example:
  python task_4.py \
    --original-dir "/path/to/cadec" \
    --gold-subdir "meddra" \
    --predicted-dir "/path/to/predicted" \
    --text-dir "/path/to/cadec/text" \
    --file "ARTHROTEC.24" \
    --show-labels \
    --match strict
"""

import argparse
import collections
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Set, Dict

# ---------- Data ----------
@dataclass(frozen=True)
class Span:
    label: str
    start: int
    end: int

RANGE_RE = re.compile(r"(\d+)\s+(\d+)")  # matches every "start end" pair (handles discontiguous ranges)

def _iter_t_lines(path: Path):
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or not line.startswith("T"):
            continue
        parts = line.split("\t")
        if len(parts) < 3:
            # need at least: ID, "Label s e[;s e]*", text
            continue
        yield parts  # [tid, head, text]

def parse_meddra_gold_as_adr(path: Path) -> List[Span]:
    """Treat all T-line spans in meddra as ADR; split multi-ranges into individual spans."""
    out: List[Span] = []
    for _, head, _ in _iter_t_lines(path):
        # head example: "10028294 486 490;496 508"  OR  "ADR 486 508"
        first_token = head.split()[0]
        tail = head[len(first_token):]
        for m in RANGE_RE.finditer(tail):
            s, e = int(m.group(1)), int(m.group(2))
            if e > s:
                out.append(Span("ADR", s, e))
    # unique & sorted
    out = sorted(set(out), key=lambda x: (x.start, x.end))
    return out

def parse_pred_adr_only(path: Path) -> List[Span]:
    """Keep only ADR-labeled spans from predictions; split multi-ranges."""
    out: List[Span] = []
    for _, head, _ in _iter_t_lines(path):
        label = head.split()[0]
        if re.sub(r"[^a-z0-9]+", "", label.lower()) != "adr":
            continue
        tail = head[len(label):]
        for m in RANGE_RE.finditer(tail):
            s, e = int(m.group(1)), int(m.group(2))
            if e > s:
                out.append(Span("ADR", s, e))
    out = sorted(set(out), key=lambda x: (x.start, x.end))
    return out

def to_set(spans: List[Span]) -> Set[Tuple[str, int, int]]:
    return {(s.label, s.start, s.end) for s in spans}

def prf1(tp: int, fp: int, fn: int):
    p = tp/(tp+fp) if tp+fp else 0.0
    r = tp/(tp+fn) if tp+fn else 0.0
    f = 2*p*r/(p+r) if p+r else 0.0
    return p, r, f

def overlap_len(a: Tuple[int, int], b: Tuple[int, int]) -> int:
    return max(0, min(a[1], b[1]) - max(a[0], b[0]))

# ---------- Pretty printing ----------
def load_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")

def with_text(spans: List[Span], raw: str):
    for s in spans:
        yield (s.label, s.start, s.end, raw[s.start:s.end].replace("\n", " "))

# ---------- Main ----------
def main():
    ap = argparse.ArgumentParser(description="ADR-only eval; gold from meddra (all T-lines treated as ADR).")
    ap.add_argument("--original-dir", required=True, help="Root CADEC dir that contains <gold-subdir>/")
    ap.add_argument("--gold-subdir", default="meddra", help="Subdir name for MedDRA gold (default: meddra)")
    ap.add_argument("--predicted-dir", required=True)
    ap.add_argument("--text-dir", required=True)
    ap.add_argument("--file", required=True, help='Basename w/o extension, e.g. "ARTHROTEC.24"')
    ap.add_argument("--show-labels", action="store_true", help="Print label histogram for debugging")
    ap.add_argument("--match", choices=["strict", "overlap"], default="strict",
                    help="strict = exact offsets; overlap = TP if any char overlap (greedy)")
    args = ap.parse_args()

    base = args.file[:-4] if args.file.lower().endswith(".ann") else args.file

    gold_ann = Path(args.original_dir) / args.gold_subdir / f"{base}.ann"
    pred_ann = Path(args.predicted_dir) / f"{base}.ann"
    txt_path = Path(args.text_dir) / f"{base}.txt"

    if not gold_ann.exists(): raise FileNotFoundError(f"Gold not found: {gold_ann}")
    if not pred_ann.exists(): raise FileNotFoundError(f"Pred not found: {pred_ann}")
    if not txt_path.exists():  raise FileNotFoundError(f"Text not found: {txt_path}")

    raw = load_text(txt_path)
    gold = parse_meddra_gold_as_adr(gold_ann)
    pred = parse_pred_adr_only(pred_ann)

    if args.show_labels:
        # Show what labels exist (useful if your predicted file contains non-ADR majority)
        import collections
        def hist(spans: List[Span]) -> Dict[str, int]:
            return dict(sorted(collections.Counter(s.label for s in spans).items()))
        print("\n[Label histogram] gold(meddra treated as ADR):", hist(gold))
        print("[Label histogram] pred (ADR only)            :", hist(pred))

    print("\n--- Ground Truth ADR Entities (meddra) ---")
    for t in with_text(gold, raw):
        print(t)

    print("\n--- Predicted ADR Entities ---")
    for t in with_text(pred, raw):
        print(t)

    if args.match == "strict":
        gset = to_set(gold)
        pset = to_set(pred)
        tp = len(gset & pset)
        fp = len(pset - gset)
        fn = len(gset - pset)
    else:
        # overlap matching: each gold can match at most one pred
        gold_unused = gold[:]
        tp = 0
        for p in pred:
            ps, pe = p.start, p.end
            hit_idx = None
            for i, g in enumerate(gold_unused):
                if overlap_len((ps, pe), (g.start, g.end)) > 0:
                    hit_idx = i; break
            if hit_idx is not None:
                tp += 1
                gold_unused.pop(hit_idx)
        fp = len(pred) - tp
        fn = len(gold_unused)

    P, R, F = prf1(tp, fp, fn)
    print("\n--- ADR-only Evaluation Metrics ---")
    print(f"TP: {tp}  FP: {fp}  FN: {fn}")
    print(f"Precision: {P:.2f}")
    print(f"Recall:    {R:.2f}")
    print(f"F1-score:  {F:.2f}")

if __name__ == "__main__":
    main()
