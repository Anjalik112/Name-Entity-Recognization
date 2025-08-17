
from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

# --- .env ---
try:
    from dotenv import load_dotenv
    load_dotenv()
    load_dotenv(dotenv_path=Path(__file__).with_name(".env"))
except Exception:
    pass

# --- Defaults (Groq only) ---
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "llama-3.1-8b-instant")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_BASEURL = os.getenv("GROQ_BASEURL", "https://api.groq.com/openai/v1")

MAX_CHARS = 1500
TIMEOUT = 60

# --- Labels & priority (Finding optional at runtime) ---
BASE_LABELS = ("ADR", "Drug", "Disease", "Symptom")
PRIORITY_BASE = {"ADR": 4, "Drug": 3, "Disease": 2, "Symptom": 1}
PRIORITY_WITH_FINDING = {"ADR": 5, "Drug": 4, "Disease": 3, "Symptom": 2, "Finding": 1}

# --- Data ---
@dataclass
class Range:
    start: int
    end: int

@dataclass
class RawSpan:
    label: str
    ranges: List[Range]
    text: str

@dataclass
class Span:
    label: str
    ranges: List[Range]

# -------------------------
# Prompt builder (policy-aware)
# -------------------------
def build_system_prompt(keep_finding: bool, allow_procedures: bool, distance_policy: str) -> str:
    labels_list = list(BASE_LABELS)
    if keep_finding:
        labels_list.append("Finding")
    labels_str = ", ".join(labels_list)

    drop_bits = [
        "- Meta phrases (e.g., 'possible side effects', 'side effects' when not the patient’s actual event).",
        "- Generic terms like 'drug', 'medicine', 'medication', 'tablet', 'pill' unless a real brand/generic product name.",
        "- Dosing schedules ('twice per day', '2x/day', 'every morning').",
        "- Negated mentions in the same clause ('without bleeding', 'no cramps').",
    ]
    if not allow_procedures:
        drop_bits.append("- Procedures/plans ('surgery', 'operation', 'procedure', 'injection', 'epidural steroid injection') unless clearly the adverse event itself.")
    if distance_policy == "drop":
        drop_bits.append("- Distances/quantities by themselves ('100 meters', '1/2 km', '10 years').")

    drop_block = "\n".join(drop_bits)

    return f"""You are a clinical annotation assistant for CADEC forum posts.
Return spans ONLY as strict JSON.

ALLOWED labels (only these): {labels_str}.

OFFSETS:
- Character offsets are 0-based, end-exclusive, and MUST be within THIS CHUNK ONLY.
- Provide one or more ranges per span for discontiguous mentions.
- Every range MUST match exact text; do not hallucinate.

WHAT TO LABEL:
- Concrete, patient-experienced clinical events or conditions.
- Prefer the most specific phrase ('lower abdominal pain' over 'pain').
- If medication/brand or dosing context is present and the text indicates a side-effect, prefer ADR over Symptom.

DO NOT LABEL (drop these):
{drop_block}

Sort spans by first range start then end. Keep output minimal and precise.

Return JSON ONLY:
{{"spans":[{{"label":"{'|'.join(labels_list)}","ranges":[{{"start":int,"end":int}}],"text":"verbatim from text"}}]}}"""

USER_TEMPLATE = """CHUNK (local offsets 0..{n}):
{chunk}

Output JSON ONLY, no commentary."""

# -------------------------
# Groq call
# -------------------------
def call_groq(model: str, system: str, user: str, temperature: float = 0.0) -> str:
    if not GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY is not set in .env")
    import requests
    url = f"{GROQ_BASEURL}/chat/completions"
    headers = {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": model,
        "temperature": temperature,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    }
    r = requests.post(url, headers=headers, json=payload, timeout=TIMEOUT)
    r.raise_for_status()
    data = r.json()
    return data["choices"][0]["message"]["content"]

# -------------------------
# Chunking
# -------------------------
def chunk_text(text: str, max_chars: int = MAX_CHARS) -> List[Tuple[int, str]]:
    chunks: List[Tuple[int, str]] = []
    i, n = 0, len(text)
    while i < n:
        j = min(i + max_chars, n)
        cut = text.rfind("\n", i, j)
        if cut == -1:
            cut = text.rfind(". ", i, j)
        if cut == -1 or cut <= i + int(0.5 * max_chars):
            cut = j
        chunks.append((i, text[i:cut]))
        i = cut
    return chunks

# -------------------------
# JSON & sanitization
# -------------------------
WHITESPACE_RE = re.compile(r"\s+")
DURATION_RE = re.compile(r"^\s*\d+\s+(years?|months?|weeks?|days?)\b", re.I)

def parse_json_strict(s: str) -> Dict[str, Any]:
    try:
        return json.loads(s)
    except Exception:
        # try to salvage a JSON object
        m = re.search(r"\{.*\}$", s.strip(), flags=re.S)
        if not m:
            m = re.search(r"\{.*\}", s, flags=re.S)
        if not m:
            raise
        return json.loads(m.group(0))

def sanitize_raw_spans(resp: Dict[str, Any], chunk_len: int, allowed_labels: List[str]) -> List[RawSpan]:
    out: List[RawSpan] = []
    for item in (resp.get("spans") or []):
        label = item.get("label")
        if label not in allowed_labels:
            continue
        text = (item.get("text") or "").strip()
        ranges: List[Range] = []
        for r in (item.get("ranges") or []):
            try:
                s = int(r["start"]); e = int(r["end"])
            except Exception:
                continue
            if not (0 <= s < e <= chunk_len):
                continue
            ranges.append(Range(s, e))
        if not ranges:
            continue
        # sort & dedupe ranges
        uniq, seen = [], set()
        for rr in sorted(ranges, key=lambda x: (x.start, x.end)):
            key = (rr.start, rr.end)
            if key in seen: continue
            seen.add(key); uniq.append(rr)
        out.append(RawSpan(label=label, ranges=uniq, text=text))
    out.sort(key=lambda sp: (sp.ranges[0].start, sp.ranges[0].end))
    return out

# -------------------------
# Offset repair
# -------------------------
def norm(s: str) -> str:
    return WHITESPACE_RE.sub(" ", s).strip().casefold()

def slice_join(chunk: str, ranges: List[Range]) -> str:
    return " ".join(chunk[r.start:r.end] for r in ranges)

def find_exact(chunk: str, needle: str) -> Optional[Tuple[int, int]]:
    if not needle:
        return None
    i = chunk.find(needle)
    if i != -1:
        return (i, i + len(needle))
    lc = chunk.casefold()
    ln = needle.casefold()
    j = lc.find(ln)
    if j != -1:
        return (j, j + len(needle))
    return None

def expand_to_word_boundaries(chunk: str, s: int, e: int) -> Tuple[int, int]:
    while s > 0 and (chunk[s-1].isalnum() or chunk[s-1] in "'-"):
        s -= 1
    n = len(chunk)
    while e < n and (chunk[e].isalnum() or chunk[e] in "'-"):
        e += 1
    return s, e

def is_plausible_text(txt: str) -> bool:
    t = txt.strip()
    if len(t) < 2: return False
    letters = sum(ch.isalpha() for ch in t)
    if letters == 0: return False
    return letters / max(1, len(t)) >= 0.3

def repair_raw_span(raw: RawSpan, chunk: str) -> Optional[Span]:
    """
    Ensure ranges match model 'text'; otherwise search or expand hull.
    """
    joined = slice_join(chunk, raw.ranges)
    if norm(joined) == norm(raw.text):
        return Span(label=raw.label, ranges=raw.ranges)

    match = find_exact(chunk, raw.text)
    if match:
        s, e = match
        s, e = expand_to_word_boundaries(chunk, s, e)
        surf = chunk[s:e]
        if is_plausible_text(surf):
            return Span(label=raw.label, ranges=[Range(s, e)])

    s = min(r.start for r in raw.ranges)
    e = max(r.end for r in raw.ranges)
    s, e = expand_to_word_boundaries(chunk, s, e)
    surf = chunk[s:e]
    if is_plausible_text(surf):
        return Span(label=raw.label, ranges=[Range(s, e)])

    return None

# -------------------------
# Global utils
# -------------------------
def to_global(spans: List[Span], base: int) -> List[Span]:
    return [Span(sp.label, [Range(base + r.start, base + r.end) for r in sp.ranges]) for sp in spans]

def span_hull(sp: Span) -> Tuple[int, int]:
    s = min(r.start for r in sp.ranges)
    e = max(r.end for r in sp.ranges)
    return s, e

def resolve_conflicts(spans: List[Span], keep_finding: bool) -> List[Span]:
    priority = PRIORITY_WITH_FINDING if keep_finding else PRIORITY_BASE
    # sort: start asc, length desc, priority desc
    spans = sorted(spans, key=lambda sp: (span_hull(sp)[0], -(span_hull(sp)[1]-span_hull(sp)[0]), -priority.get(sp.label, 0)))
    kept: List[Span] = []
    for sp in spans:
        s, e = span_hull(sp)
        conflict = False
        for kp in kept:
            ks, ke = span_hull(kp)
            if not (e <= ks or ke <= s):
                # prefer higher priority, then longer
                if priority.get(kp.label, 0) > priority.get(sp.label, 0):
                    conflict = True; break
                if priority.get(kp.label, 0) == priority.get(sp.label, 0) and (ke-ks) >= (e-s):
                    conflict = True; break
        if not conflict:
            kept.append(sp)
    # dedupe exact repeats
    uniq, seen = [], set()
    for sp in kept:
        key = (sp.label, tuple((r.start, r.end) for r in sp.ranges))
        if key in seen: continue
        seen.add(key); uniq.append(sp)
    uniq.sort(key=lambda sp: (span_hull(sp)[0], span_hull(sp)[1], -priority.get(sp.label, 0)))
    return uniq

def clip_to_text(spans: List[Span], text: str) -> List[Span]:
    n = len(text); out: List[Span] = []
    for sp in spans:
        rr = [r for r in sp.ranges if 0 <= r.start < r.end <= n]
        if rr:
            out.append(Span(sp.label, rr))
    return out

# -------------------------
# Policy-aware post-filters (universal rules)
# -------------------------
META_PHRASES_RE = re.compile(r"\b(side effect|side effects|possible side effects?)\b", re.I)
FREQUENCY_RE = re.compile(r"\b(?:once|twice|thrice|\d+\s*(?:x|times?))\s*(?:per|a)\s*(?:day|week|month|hour)s?\b", re.I)
DISTANCE_SURF_RE = re.compile(r"^\s*\d+(?:\.\d+)?\s*(?:m|km|meter|meters|kilometer|kilometers|kms?)\s*$", re.I)
PROCEDURE_RE = re.compile(r"\b(surgery|operation|procedure|injection|epidural|steroid injection)\b", re.I)
GENERIC_DRUG_SURF_RE = re.compile(r"^(?:\b(this|that|the|my|his|her)\b\s+)?\b(drug|medicine|medication|tablet|pill)s?\b$", re.I)
WEEKDAY_RE = re.compile(r"^(mon(day)?|tue(sday)?|wed(nesday)?|thu(rsday)?|fri(day)?|sat(urday)?|sun(day)?)$", re.I)
NON_EVENT_PHRASE_RE = re.compile(r"\b(this|that|the)\s+poison\b", re.I)

SYMPTOM_HEAD_RE = re.compile(
    r"\b(pain|cramps?|bleeding|nausea|vomit(?:ing)?|diarr(?:hea|hoea)|diah?rea|diarh?ea|headache|dizz(?:y|iness)|rash|swelling)\b", re.I
)
MENSTRUAL_RE = re.compile(r"\b(menstrual|menstruation|periods?|menorrhagia|vaginal bleeding|bleeding from the vagina)\b", re.I)
MENSTRUAL_EXTRA_RE = re.compile(r"\b(menstrual cramps?|vaginal cramps?|uter(?:us|ine) contractions?)\b", re.I)

# Medication context anywhere in doc
DOSAGE_RE = re.compile(r"\b\d+\s*(?:mg|mcg|g|ml|iu|units?|caps?|tabs?)\b", re.I)
MED_CUES_RE = re.compile(r"\b(took|taking|dose|dosing|tablet|pill|medication|medicine|nsaid|ibuprofen|advil|naproxen|arthrotec|drug)\b", re.I)

FUNC_VERBS = {"walk", "walking", "run", "running", "stand", "standing", "lift", "able", "unable", "can", "can't", "cannot", "limited", "limit"}

# Small drug lexicon & brand surface helpers
DRUG_LEXICON = {
    "Arthrotec","Misoprostol","Diclofenac","Ibuprofen","Paracetamol",
    "Advil","Naproxen","Voltaren","Lyrica","Lipitor","Cymbalta",
    "Clonidine","Tylenol","co-codamol","Pamprin"
}
DRUG_LEXICON_RE = re.compile(r"\b(" + "|".join(sorted(map(re.escape, DRUG_LEXICON))) + r")\b", re.I)
PROPER_BRAND_RE = re.compile(r"^[A-Z][A-Za-z0-9\-]{2,}$")

def build_span_text(text: str, ranges: List[Range]) -> str:
    return " ".join(text[r.start:r.end].replace("\n", " ").replace("\t", " ") for r in ranges)

def has_med_context(text: str) -> bool:
    return bool(DOSAGE_RE.search(text) or MED_CUES_RE.search(text))

def has_local_med_context(text: str, s: int, e: int, window: int = 120) -> bool:
    L = max(0, s - window)
    R = min(len(text), e + window)
    ctx = text[L:R]
    return bool(MED_CUES_RE.search(ctx) or DOSAGE_RE.search(ctx))

def has_functional_context(text: str, s: int, e: int, window: int = 60) -> bool:
    n = len(text)
    L = max(0, s - window)
    R = min(n, e + window)
    ctx = text[L:R].casefold()
    return any(v in ctx for v in FUNC_VERBS)

def post_filter_spans(
    spans: List[Span],
    text: str,
    keep_finding: bool,
    allow_procedures: bool,
    distance_policy: str,
    menstrual_policy: str,
    disease_symptom_relabel: str,
) -> List[Span]:
    out: List[Span] = []
    med_ctx = has_med_context(text)

    allowed_labels = set(BASE_LABELS) if not keep_finding else {"ADR", "Drug", "Disease", "Symptom", "Finding"}

    for sp in spans:
        if sp.label not in allowed_labels:
            continue

        surf = build_span_text(text, sp.ranges).strip()
        if len(surf) < 2:
            continue

        # compute hull once for context checks
        s0 = min(r.start for r in sp.ranges)
        e0 = max(r.end for r in sp.ranges)

        # Simple noise drops
        if sp.label == "Drug" and WEEKDAY_RE.fullmatch(surf):
            continue
        if sp.label == "ADR" and NON_EVENT_PHRASE_RE.search(surf):
            continue

        # Universal drops
        if META_PHRASES_RE.search(surf): 
            continue
        if FREQUENCY_RE.search(surf): 
            continue

        # Procedures
        if not allow_procedures and PROCEDURE_RE.search(surf):
            continue

        # Distances
        if distance_policy == "drop" and DISTANCE_SURF_RE.match(surf):
            continue
        elif distance_policy == "functional" and DISTANCE_SURF_RE.match(surf):
            if not has_functional_context(text, s0, e0):
                continue
            if keep_finding and sp.label != "Finding":
                sp = Span(label="Finding", ranges=sp.ranges)

        # Drug surface: drop generic mentions & very long non-brands
        if sp.label == "Drug":
            longish = ("," in surf) or (len(surf.split()) > 4)
            looks_brand = bool(DRUG_LEXICON_RE.search(surf) or PROPER_BRAND_RE.fullmatch(surf.strip()))
            dosage_near = bool(DOSAGE_RE.search(text[max(0, s0-40):min(len(text), e0+40)]))
            if GENERIC_DRUG_SURF_RE.match(surf):
                continue
            if longish and not (looks_brand or dosage_near):
                continue

        # Duration-only
        if DURATION_RE.match(surf):
            continue

        # Disease → Symptom/ADR relabel when surface looks like symptom head
        if disease_symptom_relabel == "on" and sp.label == "Disease" and SYMPTOM_HEAD_RE.search(surf):
            sp = Span(label=("ADR" if med_ctx else "Symptom"), ranges=sp.ranges)

        # Menstrual/bleeding policy (bleeding + cramps/uterus variants)
        if MENSTRUAL_RE.search(surf) or MENSTRUAL_EXTRA_RE.search(surf):
            if menstrual_policy == "adr":
                sp = Span(label="ADR", ranges=sp.ranges)
            elif menstrual_policy == "symptom":
                sp = Span(label="Symptom", ranges=sp.ranges)
            else:
                sp = Span(label=("ADR" if med_ctx else "Symptom"), ranges=sp.ranges)

        # General Symptom → ADR promotion if med context (local OR global)
        if sp.label == "Symptom" and SYMPTOM_HEAD_RE.search(surf):
            if med_ctx or has_local_med_context(text, s0, e0, window=120):
                sp = Span(label="ADR", ranges=sp.ranges)

        # ADR mis-tag that looks like a brand → flip to Drug
        if sp.label == "ADR":
            if DRUG_LEXICON_RE.search(surf) or PROPER_BRAND_RE.fullmatch(surf):
                sp = Span(label="Drug", ranges=sp.ranges)

        out.append(sp)

    return out

# -------------------------
# Enumeration splitter
# -------------------------
LIST_SEP_RE = re.compile(r",")

def _trim_to_text_bounds(text: str, s: int, e: int) -> tuple[int, int]:
    while s < e and text[s].isspace():
        s += 1
    while e > s and text[e-1].isspace():
        e -= 1
    while s > 0 and (text[s-1].isalnum() or text[s-1] in "'-"):
        s -= 1
    n = len(text)
    while e < n and (text[e].isalnum() or text[e] in "'-"):
        e += 1
    return s, e

def _looks_like_item(surf: str) -> bool:
    surf = surf.strip()
    if len(surf) < 2:
        return False
    alpha = sum(ch.isalpha() for ch in surf)
    return alpha >= 2

def split_enumerations(spans: List[Span], text: str,
                       labels_to_split: Iterable[str] = ("Symptom", "ADR"),
                       also_split_and: bool = False) -> List[Span]:
    out: List[Span] = []
    for sp in spans:
        if sp.label not in labels_to_split or len(sp.ranges) != 1:
            out.append(sp)
            continue

        r = sp.ranges[0]
        chunk = text[r.start:r.end]
        if ("," not in chunk) and not (also_split_and and " and " in chunk.lower()):
            out.append(sp)
            continue

        parts: List[tuple[int,int]] = []
        last = 0
        for m in LIST_SEP_RE.finditer(chunk):
            parts.append((last, m.start()))
            last = m.end()
        parts.append((last, len(chunk)))

        items: List[tuple[int,int]] = []
        for (ps, pe) in parts:
            sub = chunk[ps:pe]
            if also_split_and and " and " in sub.lower() and "," not in sub:
                idx = sub.lower().find(" and ")
                if idx != -1:
                    items.append((ps, ps+idx))
                    items.append((ps+idx+5, pe))
                else:
                    items.append((ps, pe))
            else:
                items.append((ps, pe))

        children: List[Span] = []
        for (ps, pe) in items:
            s = r.start + ps
            e = r.start + pe
            s, e = _trim_to_text_bounds(text, s, e)
            if e <= s:
                continue
            surf = text[s:e]
            if not _looks_like_item(surf):
                continue
            children.append(Span(sp.label, [Range(s, e)]))

        if len(children) >= 2:
            out.extend(children)
        else:
            out.append(sp)
    return out

# -------------------------
# Writer (.ann)
# -------------------------
def to_brat_ann_lines(spans: Iterable[Span], text: str) -> List[str]:
    lines: List[str] = []
    i = 1
    for sp in spans:
        coords = ";".join(f"{r.start} {r.end}" for r in sp.ranges)
        surf = build_span_text(text, sp.ranges)
        lines.append(f"T{i}\t{sp.label} {coords}\t{surf}")
        i += 1
    return lines

# -------------------------
# Orchestration
# -------------------------
def process_one_file(
    path: Path,
    model: str,
    out_dir: Path,
    temperature: float,
    max_chars: int,
    verbose: bool,
    keep_finding: bool,
    allow_procedures: bool,
    distance_policy: str,
    menstrual_policy: str,
    disease_symptom_relabel: str,
    split_enums: bool,
    split_and: bool,
) -> Path:
    raw = path.read_text(encoding="utf-8")
    chunks = chunk_text(raw, max_chars=max_chars)

    labels_list = list(BASE_LABELS)
    if keep_finding:
        labels_list.append("Finding")

    system_prompt = build_system_prompt(keep_finding, allow_procedures, distance_policy)

    all_spans: List[Span] = []
    for base, chunk in chunks:
        user = USER_TEMPLATE.format(n=len(chunk), chunk=chunk)
        resp_text = call_groq(model, system_prompt, user, temperature=temperature)
        try:
            data = parse_json_strict(resp_text)
        except Exception as ex:
            if verbose:
                print(f"[WARN] JSON parse failed @ {path.name} chunk@{base}: {ex}\n{resp_text[:400]}")
            continue
        raw_spans = sanitize_raw_spans(data, len(chunk), labels_list)

        repaired: List[Span] = []
        for rs in raw_spans:
            sp = repair_raw_span(rs, chunk)
            if sp is not None:
                repaired.append(sp)

        global_spans = to_global(repaired, base)
        all_spans.extend(global_spans)

    # clip & policy filters
    all_spans = clip_to_text(all_spans, raw)
    all_spans = post_filter_spans(
        all_spans,
        raw,
        keep_finding=keep_finding,
        allow_procedures=allow_procedures,
        distance_policy=distance_policy,
        menstrual_policy=menstrual_policy,
        disease_symptom_relabel=disease_symptom_relabel,
    )

    # split comma-separated enumerations (ADR/Symptom)
    if split_enums:
        all_spans = split_enumerations(all_spans, raw, labels_to_split=("Symptom", "ADR"), also_split_and=split_and)

    final_spans = resolve_conflicts(all_spans, keep_finding=keep_finding)
    ann_lines = to_brat_ann_lines(final_spans, raw)

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / (path.stem + ".ann")
    out_path.write_text("\n".join(ann_lines) + ("\n" if ann_lines else ""), encoding="utf-8")

    if verbose:
        print(f"\n--- {path.name} ({len(raw)} chars) ---")
        for ln in ann_lines:
            print(ln)
    print(f"[OK] {path.name}: {len(final_spans)} spans → {out_path}")
    return out_path

def process_all(
    text_dir: Path,
    model: str,
    out_dir: Path,
    **kw,
):
    files = sorted([p for p in text_dir.iterdir() if p.suffix.lower() == ".txt"])
    total, ok = 0, 0
    for p in files:
        total += 1
        try:
            out = process_one_file(p, model, out_dir, **kw)
            if out.exists() and out.stat().st_size > 0:
                ok += 1
        except Exception as ex:
            print(f"[WARN] {p.name}: {ex}")
    print(f"Done: {ok}/{total} files produced spans.")

def main():
    ap = argparse.ArgumentParser(description="CADEC → BRAT .ann (Groq-only, universal policies)")
    ap.add_argument("--text-dir", type=str, default=".", help="Directory with .txt files")
    ap.add_argument("--file", type=str, required=True, help='"all" or a filename (e.g., ARTHROTEC.17.txt)')
    ap.add_argument("--out-dir", type=str, default="./predicted", help="Output directory for .ann")
    ap.add_argument("--model", type=str, default=DEFAULT_MODEL, help="Groq model id")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max-chars", type=int, default=MAX_CHARS)
    ap.add_argument("--verbose", action="store_true")

    # Policies
    ap.add_argument("--keep-finding", action="store_true", help="Keep 'Finding' label (default: off)")
    ap.add_argument("--allow-procedures", action="store_true", help="Keep procedures/injections (default: drop)")
    ap.add_argument("--distance-policy", choices=["drop", "functional"], default="drop",
                    help="Drop distances or keep only with functional context (default: drop)")
    ap.add_argument("--menstrual-policy", choices=["auto", "adr", "symptom"], default="auto",
                    help="How to label menstrual/bleeding phrases (default: auto)")
    ap.add_argument("--disease-symptom-relabel", choices=["on", "off"], default="on",
                    help="Relabel disease→symptom/ADR when surface looks symptomatic (default: on)")

    # Enumeration splitting controls
    ap.add_argument("--split-enums", dest="split_enums", action="store_true", help="Split comma-separated ADR/Symptom lists (default)")
    ap.add_argument("--no-split-enums", dest="split_enums", action="store_false", help="Disable splitting of comma lists")
    ap.add_argument("--split-and", action="store_true", help="Also split a single ' and ' inside a list item (conservative)")
    ap.set_defaults(split_enums=True)

    args = ap.parse_args()

    text_dir = Path(args.text_dir)
    out_dir = Path(args.out_dir)

    common_kwargs = dict(
        temperature=args.temperature,
        max_chars=args.max_chars,
        verbose=args.verbose,
        keep_finding=args.keep_finding,
        allow_procedures=args.allow_procedures,
        distance_policy=args.distance_policy,
        menstrual_policy=args.menstrual_policy,
        disease_symptom_relabel=args.disease_symptom_relabel,
        split_enums=args.split_enums,
        split_and=args.split_and,
    )

    if args.file.lower() == "all":
        process_all(text_dir, args.model, out_dir, **common_kwargs)
    else:
        path = text_dir / args.file
        if not path.exists():
            raise FileNotFoundError(path)
        process_one_file(path, args.model, out_dir, **common_kwargs)

if __name__ == "__main__":
    main()
