"""
university_ranking.py
---------------------
Builds a unified university ranking lookup from three datasets:
  - CWUR  (Center for World University Rankings, 2015)
  - Shanghai/ARWU (Academic Ranking of World Universities, 2015)
  - Times Higher Education (2016)

Then maps each institution name from the resume data to a normalized
ranking score in [0, 1] (higher = better ranked).

Features produced per resume row:
  university_rank_score  : float [0, 1]  — 0 = unknown/unranked, 1 = #1 globally
  university_is_ranked   : int   {0, 1}  — whether ANY ranking system has this school
  university_world_rank  : float [0, 1]  — best (lowest) world rank, normalized to 0-1
                                           (1 = rank 1, 0 = rank 1000+)

Matching strategy (per institution name in the resume):
  1. Exact lowercase match against the master lookup
  2. Abbreviation expansion  (IIT Kanpur → Indian Institute of Technology Kanpur)
  3. Name normalization      (strip department suffixes, punctuation)
  4. Fuzzy string matching   (difflib SequenceMatcher, cutoff 0.75)

For a person with multiple institutions, we take the BEST (highest) score.
"""

import re
import ast
import pandas as pd
import numpy as np
from difflib import get_close_matches, SequenceMatcher
from typing import Optional


# ──────────────────────────────────────────────────────────────────────────────
# Abbreviation → full name expansion table
# (add entries here as needed)
# ──────────────────────────────────────────────────────────────────────────────
ABBREVIATION_MAP = {
    # Indian Institutes of Technology
    r"\biit\s+kanpur\b":       "Indian Institute of Technology Kanpur",
    r"\biit\s+kharagpur\b":    "Indian Institute of Technology Kharagpur",
    r"\biit\s+bombay\b":       "Indian Institute of Technology Bombay",
    r"\biit\s+mumbai\b":       "Indian Institute of Technology Bombay",
    r"\biit\s+delhi\b":        "Indian Institute of Technology Delhi",
    r"\biit\s+madras\b":       "Indian Institute of Technology Madras",
    r"\biit\s+roorkee\b":      "Indian Institute of Technology Roorkee",
    r"\biit\s+guwahati\b":     "Indian Institute of Technology Guwahati",
    r"\biit\s+hyderabad\b":    "Indian Institute of Technology Hyderabad",
    r"\biit\s+gandhinagar\b":  "Indian Institute of Technology Gandhinagar",
    r"\biit\s+patna\b":        "Indian Institute of Technology Patna",
    r"\biit\s+bhubaneswar\b":  "Indian Institute of Technology Bhubaneswar",
    r"\biit\b":                "Indian Institute of Technology",
    # Indian Institute of Science
    r"\biisc\b":               "Indian Institute of Science",
    r"\biis[ck]\b":            "Indian Institute of Science",
    # BITS
    r"\bbits\s+pilani\b":      "Birla Institute of Technology and Science, Pilani",
    r"\bbits\b":               "Birla Institute of Technology and Science",
    # NITs
    r"\bnit\s+warangal\b":     "National Institute of Technology Warangal",
    r"\bnit\s+trichy\b":       "National Institute of Technology Tiruchirappalli",
    r"\bnit\s+surathkal\b":    "National Institute of Technology Karnataka",
    r"\bnit\s+jaipur\b":       "National Institute of Technology Jaipur",
    r"\bnit\b":                "National Institute of Technology",
    # IIIT
    r"\biiit\s+bangalore\b":   "International Institute of Information Technology Bangalore",
    r"\biiit\s+delhi\b":       "Indraprastha Institute of Information Technology Delhi",
    r"\biiit\s+hyderabad\b":   "International Institute of Information Technology Hyderabad",
    # VIT
    r"\bvit\s+vellore\b":      "Vellore Institute of Technology",
    r"\bvit\b":                "Vellore Institute of Technology",
    # SRM
    r"\bsrm\b":                "SRM Institute of Science and Technology",
    # Penn State
    r"\bpenn\s+state\b":       "Pennsylvania State University",
    # MIT — careful not to match "AMIT" etc.
    r"^mit$":                  "Massachusetts Institute of Technology",
    # Caltech
    r"\bcaltech\b":            "California Institute of Technology",
    # UC system
    r"\buc\s+berkeley\b":      "University of California, Berkeley",
    r"\buc\s+san\s+diego\b":   "University of California, San Diego",
    r"\buc\s+los\s+angeles\b": "University of California, Los Angeles",
    r"\bucla\b":               "University of California, Los Angeles",
    r"\busc\b":                "University of Southern California",
    # Carnegie Mellon
    r"\bcmu\b":                "Carnegie Mellon University",
    # Georgia Tech
    r"\bgeorgia\s+tech\b":     "Georgia Institute of Technology",
    # Courant / NYU
    r"\bnyu\b":                "New York University",
}

# Patterns to strip from institution names before matching
STRIP_PATTERNS = [
    r",\s*(department|dept|school|college|faculty|institute|center|centre|division|program)\s+of.*$",
    r",\s*(city|state)\s*$",
    r"\s*\(.*?\)\s*",       # remove anything in parentheses
    r"\s*-\s*city\s*,\s*state\s*$",
    r"\s*-\s*state\s*$",
    r",\s*[a-z]{2}\s*$",    # trailing two-letter state code
    r"\s+campus\s*$",
    r"\s+university\s+college\s*$",  # "X university college" → keep X
]


# ──────────────────────────────────────────────────────────────────────────────
# Name normalization
# ──────────────────────────────────────────────────────────────────────────────

def _normalize_name(name: str) -> str:
    """Lowercase, strip department suffixes, expand abbreviations."""
    n = str(name).lower().strip()

    # Strip department / location suffixes
    for pat in STRIP_PATTERNS:
        n = re.sub(pat, "", n, flags=re.IGNORECASE).strip()

    # Expand abbreviations
    for pattern, expansion in ABBREVIATION_MAP.items():
        n = re.sub(pattern, expansion.lower(), n, flags=re.IGNORECASE)

    # Collapse whitespace and remove punctuation except spaces
    n = re.sub(r"[^\w\s]", " ", n)
    n = re.sub(r"\s+", " ", n).strip()
    return n


# ──────────────────────────────────────────────────────────────────────────────
# Build the unified ranking lookup
# ──────────────────────────────────────────────────────────────────────────────

def build_ranking_lookup(ranking_dir: str) -> pd.DataFrame:
    """
    Load all three ranking CSVs, take most recent year from each, normalize
    scores to [0, 1], and return a DataFrame with columns:
      canonical_name, normalized_score, world_rank_norm

    canonical_name    : lowercased / normalized institution name
    normalized_score  : average of all available system scores, 0-1 (1=best)
    world_rank_norm   : 1 - (best_rank - 1) / max_rank  (1=rank1, 0=lowest)
    """
    import os

    # ── CWUR ─────────────────────────────────────────────────────────────────
    cwur = pd.read_csv(os.path.join(ranking_dir, "cwurData.csv"))
    cwur_l = cwur[cwur["year"] == cwur["year"].max()].copy()
    # score already 44-100 → normalize to 0-1
    cwur_l["norm_score"] = cwur_l["score"] / 100.0
    cwur_l["canonical"] = cwur_l["institution"].apply(_normalize_name)
    cwur_l["world_rank_int"] = pd.to_numeric(cwur_l["world_rank"], errors="coerce")
    cwur_max_rank = cwur_l["world_rank_int"].max()

    # ── Shanghai ──────────────────────────────────────────────────────────────
    shanghai = pd.read_csv(os.path.join(ranking_dir, "shanghaiData.csv"))
    shanghai_l = shanghai[shanghai["year"] == shanghai["year"].max()].copy()
    # total_score is 0-100
    shanghai_l["norm_score"] = pd.to_numeric(shanghai_l["total_score"], errors="coerce").fillna(0) / 100.0
    shanghai_l["canonical"] = shanghai_l["university_name"].apply(_normalize_name)
    # Shanghai rank is given as "1", "2", ..., "99", "100-150" etc. — take lower bound
    def _parse_rank(r):
        s = str(r).split("-")[0]
        try:
            return int(s)
        except:
            return 500
    shanghai_l["world_rank_int"] = shanghai_l["world_rank"].apply(_parse_rank)
    shanghai_max_rank = 500   # only top 500 listed

    # ── Times ────────────────────────────────────────────────────────────────
    times = pd.read_csv(os.path.join(ranking_dir, "timesData.csv"), on_bad_lines="skip")
    times_l = times[times["year"] == times["year"].max()].copy()
    # total_score is string "95.2" etc.
    times_l["norm_score"] = pd.to_numeric(times_l["total_score"], errors="coerce").fillna(0) / 100.0
    times_l["canonical"] = times_l["university_name"].apply(_normalize_name)
    def _parse_rank_str(r):
        s = str(r).replace("=", "").split("-")[0]
        try:
            return int(s)
        except:
            return 800
    times_l["world_rank_int"] = times_l["world_rank"].apply(_parse_rank_str)
    times_max_rank = 800

    # ── Merge all three into a single lookup ──────────────────────────────────
    frames = []
    for df, system in [(cwur_l, "cwur"), (shanghai_l, "shanghai"), (times_l, "times")]:
        frames.append(df[["canonical", "norm_score", "world_rank_int"]].assign(system=system))
    merged = pd.concat(frames, ignore_index=True)

    # Group by canonical name → take mean score, min (best) rank
    lookup = merged.groupby("canonical").agg(
        normalized_score=("norm_score", "mean"),
        best_world_rank=("world_rank_int", "min"),
    ).reset_index()

    # Normalize world rank to 0-1 (1 = rank 1, 0 = rank 1000)
    max_rank = 1000
    lookup["world_rank_norm"] = (
        1.0 - (lookup["best_world_rank"].clip(1, max_rank) - 1) / (max_rank - 1)
    )

    print(f"  Ranking lookup built: {len(lookup)} unique canonical institution names")
    return lookup


# ──────────────────────────────────────────────────────────────────────────────
# Token-set matching helpers
# ──────────────────────────────────────────────────────────────────────────────

# These words appear in almost every institution name and add no discriminative
# power — strip them before comparing distinctive tokens.
_STOP_WORDS = {
    "university", "college", "of", "the", "and", "for", "in", "at",
    "a", "an", "school", "academy", "polytechnic",
}

# Entities that look like they reference a real university but are NOT
# the university itself — excluded from matching.
_NON_UNIVERSITY_PATTERNS = [
    r"\bcoursera\b", r"\bedx\b", r"\budemy\b", r"\blinkedin\b",
    r"\bhigh\s+school\b", r"\bsenior\s+secondary\b", r"\bsecondary\s+school\b",
    r"\bgrammar\s+school\b", r"\bjunior\s+college\b",
    r"\bmilitary\b", r"\barmy\b", r"\bnavy\b", r"\bair\s+force\b", r"\bnational\s+guard\b",
    r"\bjob\s+corps\b", r"\btraining\s+program\b", r"\bseminars\b",
    r"\bstate\s+of\b",   # "State of Ohio" — government entity
    r"\bgovernment\s+of\b",
]


def _is_excluded(norm_name: str) -> bool:
    """Return True if this entry should never be matched to a ranked university."""
    for pat in _NON_UNIVERSITY_PATTERNS:
        if re.search(pat, norm_name, re.IGNORECASE):
            return True
    return False


def _key_tokens(norm_name: str) -> frozenset:
    """Return the set of non-stop-word tokens from a normalized name."""
    return frozenset(t for t in norm_name.split() if t not in _STOP_WORDS and len(t) > 1)


def _token_set_score(tokens_a: frozenset, tokens_b: frozenset) -> float:
    """
    Similarity on distinctive token sets, with a penalty when the input (a) has
    many more distinctive tokens than the lookup entry (b) — prevents "Princeton
    International School of Mathematics" from matching "Princeton University".

    Metrics:
      - forward_recall : |A∩B| / |A|  — fraction of INPUT tokens that match
      - backward_recall: |A∩B| / |B|  — fraction of LOOKUP tokens that match
      - jaccard        : |A∩B| / |A∪B|

    Score = 0.3*jaccard + 0.5*backward_recall + 0.2*forward_recall
      (backward recall gets highest weight — the lookup name must be well-covered)
    """
    if not tokens_a or not tokens_b:
        return 0.0
    inter = tokens_a & tokens_b
    if not inter:
        return 0.0
    jaccard          = len(inter) / len(tokens_a | tokens_b)
    forward_recall   = len(inter) / len(tokens_a)
    backward_recall  = len(inter) / len(tokens_b)
    return 0.3 * jaccard + 0.5 * backward_recall + 0.2 * forward_recall


# ──────────────────────────────────────────────────────────────────────────────
# Match a single institution name → (score, rank_norm, is_ranked)
# ──────────────────────────────────────────────────────────────────────────────

_LOOKUP_DF: Optional[pd.DataFrame] = None
_LOOKUP_DICT: Optional[dict] = None
_CANONICAL_LIST: Optional[list] = None
_CANONICAL_TOKENS: Optional[list] = None   # pre-computed frozensets


def _load_lookup(ranking_dir: str):
    global _LOOKUP_DF, _LOOKUP_DICT, _CANONICAL_LIST, _CANONICAL_TOKENS
    if _LOOKUP_DF is None:
        _LOOKUP_DF = build_ranking_lookup(ranking_dir)
        _LOOKUP_DICT = {
            row["canonical"]: (row["normalized_score"], row["world_rank_norm"])
            for _, row in _LOOKUP_DF.iterrows()
        }
        _CANONICAL_LIST   = list(_LOOKUP_DICT.keys())
        _CANONICAL_TOKENS = [_key_tokens(c) for c in _CANONICAL_LIST]


def _match_institution(name: str, ranking_dir: str,
                        token_cutoff: float = 0.70,
                        fuzzy_cutoff: float = 0.88):
    """
    Return (norm_score, world_rank_norm, is_ranked) for a single institution.

    Matching pipeline (in order):
      1. Exact normalized match          (fastest, perfect)
      2. Token-set similarity ≥ token_cutoff  (robust to word order / suffixes)
      3. SequenceMatcher ratio ≥ fuzzy_cutoff (last resort, high threshold)
    """
    _load_lookup(ranking_dir)

    if not name or str(name).strip().lower() in ("", "n/a", "none", "nan"):
        return 0.0, 0.0, 0

    norm       = _normalize_name(name)

    # Skip known non-university entities (online platforms, military, high schools)
    if _is_excluded(norm):
        return 0.0, 0.0, 0

    inp_tokens = _key_tokens(norm)

    # ── 1. Exact match ─────────────────────────────────────────────────────
    if norm in _LOOKUP_DICT:
        s, r = _LOOKUP_DICT[norm]
        return float(s), float(r), 1

    # ── 2. Token-set match ─────────────────────────────────────────────────
    best_tok_score = 0.0
    best_tok_idx   = -1
    for i, can_tokens in enumerate(_CANONICAL_TOKENS):
        score = _token_set_score(inp_tokens, can_tokens)
        if score > best_tok_score:
            best_tok_score = score
            best_tok_idx   = i
    if best_tok_score >= token_cutoff and best_tok_idx >= 0:
        key = _CANONICAL_LIST[best_tok_idx]
        s, r = _LOOKUP_DICT[key]
        return float(s), float(r), 1

    # ── 3. High-threshold SequenceMatcher fallback ─────────────────────────
    #    (only runs if token match failed — much rarer)
    best_ratio = 0.0
    best_key   = None
    for key in _CANONICAL_LIST:
        ratio = SequenceMatcher(None, norm, key).ratio()
        if ratio > best_ratio:
            best_ratio = ratio
            best_key   = key
    if best_ratio >= fuzzy_cutoff and best_key is not None:
        s, r = _LOOKUP_DICT[best_key]
        return float(s), float(r), 1

    return 0.0, 0.0, 0


# ──────────────────────────────────────────────────────────────────────────────
# Public API: compute ranking features for a list-string of institutions
# ──────────────────────────────────────────────────────────────────────────────

def _safe_parse_list(val):
    if pd.isna(val) or str(val).strip() == "":
        return []
    try:
        result = ast.literal_eval(str(val))
        if isinstance(result, list):
            return [str(x).strip() for x in result]
        return [str(result).strip()]
    except Exception:
        return [x.strip().strip("'\"[]") for x in str(val).split(",") if x.strip()]


def compute_university_rank_features(inst_list_str: str, ranking_dir: str):
    """
    Given the raw educational_institution_name field (list-string),
    return a dict with:
      university_rank_score  : best normalized score across all listed institutions (0-1)
      university_world_rank  : best world rank norm across all listed institutions (0-1)
      university_is_ranked   : 1 if any institution is ranked, else 0
    """
    institutions = _safe_parse_list(inst_list_str)
    if not institutions:
        return {"university_rank_score": 0.0, "university_world_rank": 0.0, "university_is_ranked": 0}

    best_score = 0.0
    best_rank  = 0.0
    is_ranked  = 0
    for inst in institutions:
        score, rank_norm, ranked = _match_institution(inst, ranking_dir)
        if ranked:
            is_ranked = 1
            best_score = max(best_score, score)
            best_rank  = max(best_rank, rank_norm)

    return {
        "university_rank_score": best_score,
        "university_world_rank": best_rank,
        "university_is_ranked":  is_ranked,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Diagnostic: show match coverage for the resume dataset
# ──────────────────────────────────────────────────────────────────────────────

def diagnose_match_coverage(resume_csv: str, ranking_dir: str):
    """Print a coverage report: how many unique institutions matched."""
    _load_lookup(ranking_dir)
    raw = pd.read_csv(resume_csv)
    institutions_col = raw["educational_institution_name"].dropna()

    all_inst_names = []
    for v in institutions_col:
        all_inst_names.extend(_safe_parse_list(v))
    all_inst_names = [x for x in all_inst_names if x and x.lower() not in ("n/a","none","nan")]

    unique_names = sorted(set(all_inst_names))
    print(f"\nTotal institution mentions in resume data: {len(all_inst_names)}")
    print(f"Unique institution names: {len(unique_names)}")
    print()

    matched, unmatched = [], []
    for name in unique_names:
        score, rank_norm, is_ranked = _match_institution(name, ranking_dir)
        if is_ranked:
            matched.append((name, score, rank_norm))
        else:
            unmatched.append(name)

    pct = len(matched) / len(unique_names) * 100
    print(f"Matched (ranked):   {len(matched):4d} / {len(unique_names)} ({pct:.1f}%)")
    print(f"Unmatched:          {len(unmatched):4d} / {len(unique_names)} ({100-pct:.1f}%)")
    print()
    print("Matched institutions (top by score):")
    for name, score, rank in sorted(matched, key=lambda x: -x[1])[:20]:
        print(f"  {name:55s} score={score:.3f}  rank_norm={rank:.3f}")
    print()
    print("Unmatched institutions (sample):")
    for name in unmatched[:30]:
        print(f"  {name}")
    return matched, unmatched


# ──────────────────────────────────────────────────────────────────────────────
# CLI diagnostic
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import os
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    diagnose_match_coverage(
        resume_csv=os.path.join(base, "data", "resume_data.csv"),
        ranking_dir=os.path.join(base, "data", "university_rankings"),
    )
