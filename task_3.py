#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, re
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple, Set

@dataclass(frozen=True)
class Span:
    label: str
    start: int
    end: int

RANGE_RE = re.compile(r"(\d+)\s+(\d+)")

def parse_ann(path: Path) -> List[Span]:
    spans: List[Span] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line[0] in {"#", "A", "R"} or not line.startswith("T"):
            continue
        parts = line.split("\t")
        if len(parts) < 2:
            continue
        head = parts[1]                     # e.g. "ADR 9 19" or "ADR 9 19;29 50"
        label = head.split()[0]
        for m in RANGE_RE.finditer(head[len(label):]):
            s, e = int(m.group(1)), int(m.group(2))
            if e > s:
                spans.append(Span(label, s, e))
    # unique & sorted
    spans = sorted(set(spans), key=lambda x: (x.label, x.start, x.end))
    return spans

def to_set(spans: List[Span]) -> Set[Tuple[str,int,int]]:
    return {(s.label, s.start, s.end) for s in spans}

def prf1(tp, fp, fn):
    p = tp/(tp+fp) if tp+fp else 0.0
    r = tp/(tp+fn) if tp+fn else 0.0
    f = 2*p*r/(p+r) if p+r else 0.0
    return p, r, f

def main():
    ap = argparse.ArgumentParser(description="Pretty strict span eval for one CADEC post.")
    ap.add_argument("--original-dir", required=True)
    ap.add_argument("--predicted-dir", required=True)
    ap.add_argument("--text-dir", required=True)
    ap.add_argument("--file", required=True, help="Basename without extension, e.g. ARTHROTEC.1")
    args = ap.parse_args()

    base = args.file[:-4] if args.file.lower().endswith(".ann") else args.file
    gold_ann = Path(args.original_dir) / f"{base}.ann"
    pred_ann = Path(args.predicted_dir) / f"{base}.ann"
    txt_path = Path(args.text_dir) / f"{base}.txt"

    raw = txt_path.read_text(encoding="utf-8")
    gold = parse_ann(gold_ann)
    pred = parse_ann(pred_ann)

    def with_text(spans: List[Span]):
        out = []
        for s in spans:
            seg = raw[s.start:s.end].replace("\n", " ")
            out.append((s.label, s.start, s.end, seg))
        return out

    gold_t = with_text(gold)
    pred_t = with_text(pred)

    # Print exactly like the screenshot
    print("\n--- Ground Truth Entities ---")
    for t in gold_t:
        print(t)

    print("\n--- Predicted Entities ---")
    for t in pred_t:
        print(t)

    gset, pset = to_set(gold), to_set(pred)
    tp = len(gset & pset)
    fp = len(pset - gset)
    fn = len(gset - pset)
    P, R, F = prf1(tp, fp, fn)

    print("\n--- Evaluation Metrics ---")
    print(f"Precision: {P:.2f}")
    print(f"Recall:    {R:.2f}")
    print(f"F1-score:  {F:.2f}")

if __name__ == "__main__":
    main()
