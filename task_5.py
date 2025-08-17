#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Task 5 — Evaluate on 50 random forum posts using Task 3's logic.

What it does:
- Randomly samples K files from cadec/text (default K=50, reproducible via --seed)
- For each sampled file, compares predicted spans (predicted/<name>.ann) vs gold (original/<name>.ann)
  * strict character-offset match
  * evaluates by label: ADR, Drug, Disease, Symptom
- Prints per-file results and aggregates:
  * micro Precision/Recall/F1 overall
  * per-label Precision/Recall/F1
- Optionally: generate missing predictions by calling your Task-2 script.

Usage example:
  python task_5.py \
    --text-dir "/path/to/cadec/text" \
    --original-dir "/path/to/cadec/original" \
    --predicted-dir "/path/to/predicted" \
    --k 50 --seed 42

If you want it to auto-generate predictions for missing files:
  python task_5.py \
    --text-dir ".../cadec/text" \
    --original-dir ".../cadec/original" \
    --predicted-dir ".../predicted" \
    --k 50 --seed 42 \
    --generate-preds \
    --task2-script "/path/to/task_2.py" \
    --hf-model "d4data/biomedical-ner-all"
"""

from __future__ import annotations

import argparse
import random
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Set

LABELS = ("ADR", "Drug", "Disease", "Symptom")

@dataclass(frozen=True)
class Span:
    label: str
    start: int
    end: int

# ----------------------
# Parsers (same rules as Task 3)
# ----------------------

def parse_original_file(path: Path) -> List[Span]:
    """Parse gold from cadec/original/<file>.ann style."""
    spans: List[Span] = []
    if not path.exists():
        return spans
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or not line.startswith("T"):
            continue
        # Format: T{n}\t{LABEL} {start} {end}[; start end ...]\t{text}
        parts = line.split("\t")
        if len(parts) < 2:
            continue
        head = parts[1]
        bits = head.split()
        if not bits:
            continue
        label = bits[0]
        # collect all number pairs after the label
        nums = [int(x) for x in bits[1:] if x.isdigit()]
        for i in range(0, len(nums), 2):
            if i + 1 >= len(nums):
                break
            s, e = nums[i], nums[i+1]
            if e > s and label in LABELS:
                spans.append(Span(label, s, e))
    # dedupe
    return sorted(set(spans), key=lambda x: (x.start, x.end, x.label))

def parse_predicted_file(path: Path) -> List[Span]:
    """Parse predictions from predicted/<file>.ann (same format as original)."""
    return parse_original_file(path)

# ----------------------
# Metrics
# ----------------------

def prf1(tp: int, fp: int, fn: int) -> Tuple[float, float, float]:
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f = (2*p*r / (p+r)) if (p + r) else 0.0
    return p, r, f

def to_set(spans: List[Span], label: str | None = None) -> Set[Tuple[int, int, str]]:
    if label is None:
        return {(s.start, s.end, s.label) for s in spans}
    return {(s.start, s.end, s.label) for s in spans if s.label == label}

# ----------------------
# Optional: generate predictions (calls your Task 2 script once per file)
# ----------------------

def ensure_prediction(
    task2_script: Path,
    hf_model: str,
    text_dir: Path,
    file_txt: Path,
    out_dir: Path,
    extra_args: List[str],
):
    """Call Task 2 to create predicted/<name>.ann if missing."""
    cmd = [
        "python", str(task2_script),
        "--text-dir", str(text_dir),
        "--file", file_txt.name,           # e.g., ARTHROTEC.1.txt
        "--model", hf_model,
        "--out-dir", str(out_dir),
    ] + extra_args
    subprocess.run(cmd, check=True)

# ----------------------
# Main evaluation
# ----------------------

def eval_file(base_name: str, gold_dir: Path, pred_dir: Path) -> Dict:
    """Evaluate one file (by base name, without extension)."""
    gold = parse_original_file(gold_dir / f"{base_name}.ann")
    pred = parse_predicted_file(pred_dir / f"{base_name}.ann")

    gset_all = to_set(gold)
    pset_all = to_set(pred)

    # overall
    tp_all = len(gset_all & pset_all)
    fp_all = len(pset_all - gset_all)
    fn_all = len(gset_all - pset_all)
    P_all, R_all, F_all = prf1(tp_all, fp_all, fn_all)

    # per label
    per_label = {}
    for lab in LABELS:
        gset = to_set(gold, lab)
        pset = to_set(pred, lab)
        tp = len(gset & pset)
        fp = len(pset - gset)
        fn = len(gset - pset)
        P, R, F = prf1(tp, fp, fn)
        per_label[lab] = dict(tp=tp, fp=fp, fn=fn, P=P, R=R, F=F, gold=len(gset), pred=len(pset))

    return dict(
        base=base_name,
        overall=dict(tp=tp_all, fp=fp_all, fn=fn_all, P=P_all, R=R_all, F=F_all,
                     gold=len(gset_all), pred=len(pset_all)),
        per_label=per_label
    )

def main():
    ap = argparse.ArgumentParser(description="Task 5: Evaluate on 50 random posts using Task 3 metrics.")
    ap.add_argument("--text-dir", required=True, help="Path to cadec/text")
    ap.add_argument("--original-dir", required=True, help="Path to cadec/original (gold)")
    ap.add_argument("--predicted-dir", required=True, help="Path to predicted anns from Task 2")
    ap.add_argument("--k", type=int, default=50, help="Number of random files to evaluate")
    ap.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    # optional auto-prediction
    ap.add_argument("--generate-preds", action="store_true", help="Generate missing predictions via Task 2")
    ap.add_argument("--task2-script", default="", help="Path to your Task 2 script (required if --generate-preds)")
    ap.add_argument("--hf-model", default="d4data/biomedical-ner-all", help="HF model to use if generating predictions")
    ap.add_argument("--task2-extra", default="--merge-gap 0 --heuristic-drug --adr-mode convert",
                    help="Extra args to pass to Task 2 when generating predictions")
    args = ap.parse_args()

    text_dir = Path(args.text_dir)
    gold_dir = Path(args.original_dir)
    pred_dir = Path(args.predicted_dir)
    task2_script = Path(args.task2_script) if args.task2_script else None
    extra_args = args.task2_extra.split() if args.task2_extra else []

    # sample K files
    all_txt = sorted([p for p in text_dir.iterdir() if p.suffix.lower() == ".txt"])
    if len(all_txt) == 0:
        raise FileNotFoundError(f"No .txt files in {text_dir}")
    rnd = random.Random(args.seed)
    sample = rnd.sample(all_txt, k=min(args.k, len(all_txt)))

    # ensure predictions if requested
    if args.generate_preds:
        if not task2_script or not task2_script.exists():
            raise FileNotFoundError("You passed --generate-preds but --task2-script is missing or invalid.")
        pred_dir.mkdir(parents=True, exist_ok=True)
        for file_txt in sample:
            base = file_txt.stem  # e.g., ARTHROTEC.1
            pred_path = pred_dir / f"{base}.ann"
            if not pred_path.exists():
                ensure_prediction(task2_script, args.hf_model, text_dir, file_txt, pred_dir, extra_args)

    # evaluate
    rows = []
    totals_overall = dict(tp=0, fp=0, fn=0, gold=0, pred=0)
    totals_label = {lab: dict(tp=0, fp=0, fn=0, gold=0, pred=0) for lab in LABELS}

    print(f"\n=== Task 5: Evaluating {len(sample)} randomly selected files (seed={args.seed}) ===")
    for i, file_txt in enumerate(sample, 1):
        base = file_txt.stem
        gold_path = gold_dir / f"{base}.ann"
        pred_path = pred_dir / f"{base}.ann"
        if not gold_path.exists():
            print(f"  [{i:02d}] SKIP {base} — missing gold: {gold_path}")
            continue
        if not pred_path.exists():
            print(f"  [{i:02d}] SKIP {base} — missing prediction: {pred_path}")
            continue

        res = eval_file(base, gold_dir, pred_dir)
        o = res["overall"]
        print(f"  [{i:02d}] {base}: P={o['P']:.3f} R={o['R']:.3f} F1={o['F']:.3f}  (gold={o['gold']}, pred={o['pred']})")

        # aggregate overall
        totals_overall["tp"] += o["tp"]
        totals_overall["fp"] += o["fp"]
        totals_overall["fn"] += o["fn"]
        totals_overall["gold"] += o["gold"]
        totals_overall["pred"] += o["pred"]

        # aggregate per-label
        for lab in LABELS:
            pl = res["per_label"][lab]
            totals_label[lab]["tp"] += pl["tp"]
            totals_label[lab]["fp"] += pl["fp"]
            totals_label[lab]["fn"] += pl["fn"]
            totals_label[lab]["gold"] += pl["gold"]
            totals_label[lab]["pred"] += pl["pred"]

        rows.append((base, o["P"], o["R"], o["F"]))

    # micro overall
    P, R, F = prf1(totals_overall["tp"], totals_overall["fp"], totals_overall["fn"])
    print("\n--- Micro (all labels) over evaluated files ---")
    print(f"Gold spans: {totals_overall['gold']} | Pred spans: {totals_overall['pred']}")
    print(f"TP: {totals_overall['tp']} | FP: {totals_overall['fp']} | FN: {totals_overall['fn']}")
    print(f"Micro Precision: {P:.4f}")
    print(f"Micro Recall   : {R:.4f}")
    print(f"Micro F1       : {F:.4f}")

    # per-label micro (aggregate)
    print("\n--- Per-label micro ---")
    for lab in LABELS:
        t = totals_label[lab]
        p, r, f = prf1(t["tp"], t["fp"], t["fn"])
        print(f"{lab:<8} P={p:.4f}  R={r:.4f}  F1={f:.4f}  (tp={t['tp']}, fp={t['fp']}, fn={t['fn']}, gold={t['gold']}, pred={t['pred']})")

if __name__ == "__main__":
    main()
