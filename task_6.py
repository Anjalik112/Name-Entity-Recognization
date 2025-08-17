#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Task 6 — Code assignment for ADRs (string-match vs embedding-match)

For a single filename (same basename) combine:
- original/<name>.ann  : ground-truth labels & spans
- sct/<name>.ann       : standard codes (SNOMED CT in CADEC-sct) & mapped terms
- text/<name>.txt      : raw text
- predicted/<name>.ann : outputs from Task 2

Build a joined structure: {code, standard_text, label_type, gt_text, gt_ranges}
Then, for each ADR in predicted/<name>.ann, assign code+text in two ways:
  (a) approximate string match (RapidFuzz if available, fallback to difflib)
  (b) embedding match (SentenceTransformers), cosine similarity

Outputs a comparison table and agreement summary.

Usage:
  python task_6.py \
    --text-dir /path/to/cadec/text \
    --original-dir /path/to/cadec/original \
    --sct-dir /path/to/cadec/sct \
    --predicted-dir /path/to/predicted \
    --file ARTHROTEC.24

Optional:
  --embed-model sentence-transformers/all-MiniLM-L6-v2
  --topn 3
  --min-fuzzy 60
  --min-cos 0.35
"""

from __future__ import annotations

import argparse
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# ------------- Data classes -------------
@dataclass(frozen=True)
class Range:
    start: int
    end: int

@dataclass
class AnnSpan:
    label: str              # ADR, Drug, Disease, Symptom (from original or predicted)
    ranges: List[Range]
    text: str               # pulled from raw text

@dataclass
class SctSpan:
    code: str               # SNOMED CT code (or MedDRA if your sct dir has those)
    ranges: List[Range]
    text: str               # mapped term from sct file (we treat this as "standard_text")

@dataclass
class Joined:
    code: str
    standard_text: str      # sct span text (mapped term)
    label_type: str         # from original ann (ADR/Drug/Disease/Symptom)
    gt_text: str            # from original ann
    gt_ranges: List[Range]

# ------------- Parsing utils -------------
RANGE_RE = re.compile(r"(\d+)\s+(\d+)")

def _read_text(p: Path) -> str:
    return p.read_text(encoding="utf-8")

def _surface(text: str, ranges: List[Range]) -> str:
    return " ".join(text[r.start:r.end].replace("\n", " ") for r in ranges)

def _parse_ann_spans(path: Path, raw: str, accept_labels=None) -> List[AnnSpan]:
    """Parse T-lines like: T1\tADR 10 20;30 40\ttext..."""
    spans: List[AnnSpan] = []
    if not path.exists():
        return spans
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or not line.startswith("T") or line.startswith("TT"):
            continue
        parts = line.split("\t")
        if len(parts) < 2:
            continue
        head = parts[1]  # "LABEL s e; s e ..."
        label = head.split()[0]
        if accept_labels and label not in accept_labels:
            continue
        rr: List[Range] = []
        for m in RANGE_RE.finditer(head[len(label):]):
            s, e = int(m.group(1)), int(m.group(2))
            if e > s:
                rr.append(Range(s, e))
        if not rr:
            continue
        txt = _surface(raw, rr)
        spans.append(AnnSpan(label=label, ranges=rr, text=txt))
    return spans

def _parse_sct_spans(path: Path, raw: str) -> List[SctSpan]:
    """
    Parse TT-lines like: TT1\t<CODE> <start> <end>[; ...]\t<mapped term or surface>
    """
    out: List[SctSpan] = []
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or not line.startswith("TT"):
            continue
        parts = line.split("\t")
        if len(parts) < 2:
            continue
        head = parts[1].strip()
        # head: "<CODE> s e [; s e ...]"
        bits = head.split()
        code = bits[0]
        rr: List[Range] = []
        for m in RANGE_RE.finditer(head[len(code):]):
            s, e = int(m.group(1)), int(m.group(2))
            if e > s:
                rr.append(Range(s, e))
        if not rr:
            continue
        # standard_text: prefer the third column if present (mapped term); else use surface
        mapped = parts[2].strip() if len(parts) >= 3 else ""
        if not mapped:
            mapped = _surface(raw, rr)
        out.append(SctSpan(code=code, ranges=rr, text=mapped))
    return out

def _overlap_len(a: Range, b: Range) -> int:
    return max(0, min(a.end, b.start + (b.end - b.start)) - max(a.start, b.start))

def _spans_overlap_len(ar: List[Range], br: List[Range]) -> int:
    tot = 0
    for a in ar:
        for b in br:
            s = max(a.start, b.start)
            e = min(a.end, b.end)
            if e > s:
                tot += (e - s)
    return tot

# ------------- Join original ↔ sct by overlap -------------
def build_joined(original_spans: List[AnnSpan], sct_spans: List[SctSpan]) -> List[Joined]:
    out: List[Joined] = []
    for g in original_spans:
        # find best sct by total overlap
        best: Optional[SctSpan] = None
        best_ol = 0
        for t in sct_spans:
            ol = _spans_overlap_len(g.ranges, t.ranges)
            if ol > best_ol:
                best_ol = ol
                best = t
        if best and best_ol > 0:
            out.append(Joined(
                code=best.code,
                standard_text=best.text,
                label_type=g.label,
                gt_text=g.text,
                gt_ranges=g.ranges
            ))
        else:
            # keep a placeholder with empty code if no match
            out.append(Joined(
                code="",
                standard_text="",
                label_type=g.label,
                gt_text=g.text,
                gt_ranges=g.ranges
            ))
    return out

# ------------- Fuzzy & Embedding matching -------------
def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip().casefold()

def fuzzy_score(a: str, b: str) -> float:
    """Return 0..100. Uses rapidfuzz if available, else difflib."""
    try:
        from rapidfuzz import fuzz
        return float(fuzz.token_set_ratio(a, b))
    except Exception:
        import difflib
        return 100.0 * difflib.SequenceMatcher(None, _norm(a), _norm(b)).ratio()

def embed_model_loader(model_name: str):
    try:
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer(model_name)
    except Exception as ex:
        raise RuntimeError(
            f"Embedding model '{model_name}' not available. "
            f"Install: pip install sentence-transformers\nDetails: {ex}"
        )

def embed_vectors(model, texts: List[str]):
    # model.encode returns np.ndarray; keep it lightweight
    return model.encode(texts, normalize_embeddings=True, convert_to_numpy=True)

def cosine_sim(u, v) -> float:
    # u, v are L2-normalized; cosine = dot
    import numpy as np
    return float(np.dot(u, v))

# ------------- Pretty print helpers -------------
def fmt_ranges(rr: List[Range]) -> str:
    return ";".join(f"{r.start}-{r.end}" for r in rr)

def print_table(rows: List[List[str]]):
    if not rows:
        return
    widths = [max(len(str(row[i])) for row in rows) for i in range(len(rows[0]))]
    for i, row in enumerate(rows):
        line = " | ".join(str(row[j]).ljust(widths[j]) for j in range(len(row)))
        print(line)
        if i == 0:
            print("-+-".join("-" * w for w in widths))

# ------------- Main orchestration -------------
def main():
    ap = argparse.ArgumentParser(description="Task 6: Map ADRs to standard codes (fuzzy vs embedding)")
    ap.add_argument("--text-dir", required=True)
    ap.add_argument("--original-dir", required=True)
    ap.add_argument("--sct-dir", required=True)
    ap.add_argument("--predicted-dir", required=True)
    ap.add_argument("--file", required=True, help="Basename with or without .txt/.ann (e.g., ARTHROTEC.24)")
    ap.add_argument("--embed-model", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--topn", type=int, default=3)
    ap.add_argument("--min-fuzzy", type=float, default=60.0)
    ap.add_argument("--min-cos", type=float, default=0.35)
    args = ap.parse_args()

    base = args.file
    base = base[:-4] if base.lower().endswith(".txt") or base.lower().endswith(".ann") else base

    text_path = Path(args.text_dir) / f"{base}.txt"
    orig_path = Path(args.original_dir) / f"{base}.ann"
    sct_path  = Path(args.sct_dir) / f"{base}.ann"
    pred_path = Path(args.predicted_dir) / f"{base}.ann"

    if not text_path.exists():
        raise FileNotFoundError(text_path)
    raw = _read_text(text_path)

    # Parse
    original_spans = _parse_ann_spans(orig_path, raw, accept_labels=None)  # include all labels
    sct_spans      = _parse_sct_spans(sct_path, raw)
    predicted_spans = _parse_ann_spans(pred_path, raw, accept_labels={"ADR"})

    # Join original ↔ sct by overlap
    joined = build_joined(original_spans, sct_spans)

    # Build candidate catalog for ADR only (from joined)
    catalog = [(j.code, j.standard_text, j.gt_text, j.label_type) for j in joined if j.label_type == "ADR" and j.code]
    if not catalog:
        print("No ADR-coded entries found in sct↔original join for this file.")
    else:
        print(f"[Info] ADR catalog size: {len(catalog)}")

    # Prepare embedding model & vectors for catalog standard_text
    emb_model = None
    emb_cat = None
    if catalog:
        try:
            emb_model = embed_model_loader(args.embed_model)
            emb_cat = embed_vectors(emb_model, [c[1] for c in catalog])  # standard_text embeddings
        except Exception as ex:
            print(f"[WARN] Embedding model unavailable: {ex}")
            emb_model = None

    # Compare for each predicted ADR
    header = [
        "Pred ADR text",
        "Fuzzy code", "Fuzzy std text", "Fuzzy score",
        "Embed code", "Embed std text", "Cosine"
    ]
    rows = [header]
    agree = 0
    total = 0

    for p in predicted_spans:
        total += 1
        ptxt = p.text

        # (a) fuzzy top-1
        best_f = (-1.0, "", "")  # (score, code, std_text)
        for code, std_text, gt_text, lab in catalog:
            sc = fuzzy_score(ptxt, std_text)
            if sc > best_f[0]:
                best_f = (sc, code, std_text)
        fuzzy_code, fuzzy_txt, fuzzy_sc = "", "", 0.0
        if best_f[0] >= args.min_fuzzy:
            fuzzy_sc, fuzzy_code, fuzzy_txt = best_f[0], best_f[1], best_f[2]

        # (b) embedding top-1
        embed_code, embed_txt, embed_cos = "", "", 0.0
        if emb_model is not None and emb_cat is not None and len(catalog) > 0:
            import numpy as np
            v = embed_vectors(emb_model, [ptxt])[0]  # shape (d,)
            sims = emb_cat @ v  # cosine because normalized
            idx = int(np.argmax(sims))
            cs = float(sims[idx])
            if cs >= args.min_cos:
                embed_code, embed_txt, embed_cos = catalog[idx][0], catalog[idx][1], cs

        rows.append([
            ptxt,
            fuzzy_code, fuzzy_txt, f"{fuzzy_sc:.1f}",
            embed_code, embed_txt, f"{embed_cos:.3f}"
        ])

        if fuzzy_code and embed_code and (fuzzy_code == embed_code):
            agree += 1

    print("\n=== ADR Code Assignment (Predicted ADRs) ===")
    print_table(rows)

    if total > 0:
        print(f"\nAgreement (fuzzy vs embedding) on assigned code: {agree}/{total} = {agree/total:.2%}")
    else:
        print("\nNo predicted ADR spans found for this file.")

    # Also show the joined catalog for transparency
    print("\n=== Joined catalog (original ↔ sct) for this file ===")
    cat_rows = [["Code", "Standard Text (from sct)", "Label", "GT Text", "GT Ranges"]]
    for code, std_text, gt_text, lab in catalog:
        cat_rows.append([code, std_text, lab, gt_text, ""])
    print_table(cat_rows)

if __name__ == "__main__":
    main()
