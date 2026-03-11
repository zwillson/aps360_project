"""
company_features.py
--------------------
Matches `professional_company_names` from resume data against the Fortune 100
companies in Largest_Companies.csv and extracts three numerical signal features.

Features produced (per row):
    company_is_fortune100      : 1 if ANY company in the candidate's history
                                 matched a Fortune 100 firm, else 0
    company_fortune100_rank_norm: best matched company's rank normalized to [0, 1]
                                 (1.0 = rank #1 / Walmart, 0.0 = unranked).
                                 Formula: 1 - (rank-1) / 99
    company_size_norm          : log-normalized employee count of the best-matched
                                 company (log10 scale, normalized to [0, 1] using
                                 the max in the reference list ≈ log10(2_100_000)).
                                 0.0 if unranked.

Matching strategy
-----------------
1. Canonical normalization:
   - Lower-case, strip punctuation, remove stop-words
   - Expand common aliases (Google → Alphabet, UPS → United Parcel Service, etc.)
2. Token-set similarity: score = 0.4×Jaccard + 0.3×forward_recall + 0.3×backward_recall
   - forward_recall  = |intersection| / |company_tokens|  (prevents over-match)
   - backward_recall = |intersection| / |resume_tokens|   (prevents under-match)
3. Accept match if score >= MATCH_THRESHOLD (0.50)
4. For each candidate, take the BEST (highest-scoring) match across all companies listed.

Placeholder filtering
---------------------
Raw data often contains "Company Name", "N/A", "Company Name ï¼ City , State"
as stand-in values. These are detected and excluded before matching.
"""

import os
import re
import ast
import math
import functools

import numpy as np
import pandas as pd


# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

MATCH_THRESHOLD = 0.50   # minimum token-set score to accept a match

# Stop-words stripped from BOTH the resume name and the reference name
_STOP_WORDS = {
    "inc", "ltd", "llc", "corp", "corporation", "company", "co", "group",
    "holdings", "services", "global", "international", "the", "and", "of",
    "for", "in", "a", "an", "at", "by", "with", "pvt", "limited", "plc",
    "sa", "ag", "bv", "se", "nv",
}

# Common aliases: map alternative names → canonical Fortune 100 company name
# Keys are lowercased, punctuation-stripped
_ALIASES = {
    "google":                   "alphabet",
    "google inc":               "alphabet",
    "google llc":               "alphabet",
    "youtube":                  "alphabet",
    "deepmind":                 "alphabet",
    "waymo":                    "alphabet",
    "ups":                      "united parcel service",
    "united parcel service":    "united parcel service",
    "facebook":                 "meta platforms",
    "facebook inc":             "meta platforms",
    "instagram":                "meta platforms",
    "whatsapp":                 "meta platforms",
    "meta":                     "meta platforms",
    "meta inc":                 "meta platforms",
    "verizon":                  "verizon communications",
    "ge":                       "general electric",
    "gm":                       "general motors",
    "j&j":                      "johnson johnson",
    "jnj":                      "johnson johnson",
    "jpmorgan":                 "jpmorgan chase",
    "jp morgan":                "jpmorgan chase",
    "jpmorgan chase":           "jpmorgan chase",
    "bank of america":          "bank of america",
    "bofa":                     "bank of america",
    "boa":                      "bank of america",
    "fedex":                    "fedex",
    "fedex corporation":        "fedex",
    "fed ex":                   "fedex",
    "merck":                    "merck co",
    "exxon":                    "exxon mobil",
    "exxonmobil":               "exxon mobil",
    "ibm":                      "ibm",
    "international business machines": "ibm",
    "chevron":                  "chevron corporation",
    "at&t":                     "att",
    "att":                      "att",
    "att inc":                  "att",
    "home depot":               "the home depot",
    "the home depot":           "the home depot",
    "american express":         "american express",
    "amex":                     "american express",
    "disney":                   "the walt disney company",
    "walt disney":              "the walt disney company",
    "procter and gamble":       "procter gamble",
    "p&g":                      "procter gamble",
    "pg":                       "procter gamble",
    "comcast":                  "comcast",
    "lockheed":                 "lockheed martin",
    "caterpillar":              "caterpillar",
    "cat":                      "caterpillar",
    "abbvie":                   "abbvie",
    "dow":                      "dow chemical company",
    "dow chemical":             "dow chemical company",
    "qualcomm":                 "qualcomm",
    "nike inc":                 "nike",
    "costco":                   "costco",
    "costco wholesale":         "costco",
    "wells fargo":              "wells fargo",
    "thermo fisher":            "thermo fisher scientific",
    "bristol myers squibb":     "bristolmyers squibb",
    "bms":                      "bristolmyers squibb",
    "delta airlines":           "delta air lines",
    "delta air":                "delta air lines",
    "united airlines":          "united airlines",
    "american airlines":        "american airlines",
    "charter":                  "charter communications",
    "sysco":                    "sysco",
    "goldman":                  "goldman sachs",
    "morgan stanley":           "morgan stanley",
    "cigna":                    "cigna",
    "humana":                   "humana",
    "aig":                      "aig",
    "metlife":                  "metlife",
    "prudential":               "prudential financial",
    "tyson":                    "tyson foods",
    "john deere":               "john deere",
    "deere":                    "john deere",
    "target":                   "target corporation",
    "kroger":                   "kroger",
    "publix":                   "publix",
    "albertsons":               "albertsons",
    "best buy":                 "best buy",
    "lowes":                    "lowes",
    "tjx":                      "tjx",
    "pfizer":                   "pfizer",
    "abbott":                   "abbott",    # not in list but common
    "walmart":                  "walmart",
    "wal mart":                 "walmart",
    "walgreens":                "walgreens boots alliance",
    "cvs":                      "cvs health",
    "cvs pharmacy":             "cvs health",
    "anthem":                   "elevance health",
    "elevance":                 "elevance health",
    "unitedhealth":             "unitedhealth group",
    "united health":            "unitedhealth group",
    "hca":                      "hca healthcare",
    "progressive":              "progressive corporation",
    "allstate":                 "allstate",
    "nationwide":               "nationwide mutual insurance company",
    "liberty mutual":           "liberty mutual",
    "state farm":               "state farm",
    "new york life":            "new york life insurance company",
    "fannie mae":               "fannie mae",
    "freddie mac":              "freddie mac",
    "conoco":                   "conocophillips",
    "conocophillips":           "conocophillips",
    "phillips66":               "phillips 66",
    "phillips 66":              "phillips 66",
    "valero":                   "valero energy",
    "marathon":                 "marathon petroleum",
    "td synnex":                "td synnex",
    "synnex":                   "td synnex",
    "mckesson":                 "mckesson corporation",
    "cardinal health":          "cardinal health",
    "amerisourcebergen":        "amerisourcebergen",
    "amerisource":              "amerisourcebergen",
    "centene":                  "centene",
    "enterprise products":      "enterprise products",
    "energy transfer":          "energy transfer partners",
    "plains":                   "plains all american pipeline",
    "bunge":                    "bunge limited",
    "chs":                      "chs",
    "performance food":         "performance food group",
    "pbf energy":               "pbf energy",
    "stoneх":                   "stonex group",
    "stonex":                   "stonex group",
    "adc":                      "archer daniels midland",
    "adm":                      "archer daniels midland",
    "archer daniels":           "archer daniels midland",
    "rtx":                      "rtx corporation",
    "raytheon":                 "rtx corporation",
    "berkshire hathaway":       "berkshire hathaway",
    "berkshire":                "berkshire hathaway",
    "world fuel":               "world fuel services",
}

# Patterns that indicate a placeholder / anonymous entry (regex, case-insensitive)
_PLACEHOLDER_PATTERNS = [
    r"^company\s*name",
    r"^n/?a$",
    r"^none$",
    r"^-+$",
    r"^unknown$",
    r"^not\s+applicable$",
    r"valve\s+value\s+stream",  # literal placeholder in this dataset
    r"ï¼",                      # encoding artifact placeholder
]
_PLACEHOLDER_RE = re.compile(
    "|".join(_PLACEHOLDER_PATTERNS), re.IGNORECASE
)

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _safe_parse_list(val) -> list:
    """Parse a Python-list string like \"['A', 'B']\" into a Python list."""
    if pd.isna(val) or str(val).strip() == "":
        return []
    try:
        result = ast.literal_eval(str(val))
        if isinstance(result, list):
            return [str(x).strip() for x in result]
        return [str(result).strip()]
    except Exception:
        return [x.strip().strip("'\"[]") for x in str(val).split(",") if x.strip()]


def _is_placeholder(name: str) -> bool:
    """Return True if the company name is a placeholder / filler entry."""
    name = name.strip()
    if not name or len(name) < 2:
        return True
    return bool(_PLACEHOLDER_RE.search(name))


def _normalize_name(name: str) -> str:
    """Lower-case, strip punctuation, collapse whitespace."""
    name = name.lower().strip()
    name = re.sub(r"[^\w\s]", " ", name)
    name = re.sub(r"\s+", " ", name).strip()
    return name


def _tokenize(name: str) -> frozenset:
    """Normalize then tokenize, removing stop-words."""
    norm = _normalize_name(name)
    # Apply alias mapping if exact match
    norm = _ALIASES.get(norm, norm)
    tokens = frozenset(norm.split()) - _STOP_WORDS
    return tokens


def _token_set_score(tokens_a: frozenset, tokens_b: frozenset) -> float:
    """
    Compute similarity: 0.4×Jaccard + 0.3×forward_recall + 0.3×backward_recall.
    forward_recall  = |inter| / |tokens_b|  (prevents partial over-match from b side)
    backward_recall = |inter| / |tokens_a|  (prevents partial over-match from a side)
    """
    if not tokens_a or not tokens_b:
        return 0.0
    inter = tokens_a & tokens_b
    union = tokens_a | tokens_b
    n_inter = len(inter)
    jaccard  = n_inter / len(union)
    fwd      = n_inter / len(tokens_b)   # recall over b (reference)
    bwd      = n_inter / len(tokens_a)   # recall over a (query)
    return 0.4 * jaccard + 0.3 * fwd + 0.3 * bwd


# ──────────────────────────────────────────────────────────────────────────────
# Reference data loader (cached per process to avoid re-reading the CSV)
# ──────────────────────────────────────────────────────────────────────────────

@functools.lru_cache(maxsize=1)
def _load_fortune_lookup(csv_path: str) -> tuple:
    """
    Load Largest_Companies.csv and return:
        lookup : list of (normalized_tokens, row_dict)  sorted by rank
        max_log_employees : float, for normalization
    """
    df = pd.read_csv(csv_path)

    # Clean employee counts (stored as "2,100,000" strings)
    def _parse_employees(v):
        try:
            return int(str(v).replace(",", ""))
        except Exception:
            return 0

    df["_employees_int"] = df["Employees"].apply(_parse_employees)

    max_log_emp = math.log10(max(df["_employees_int"].max(), 1))

    lookup = []
    for _, row in df.iterrows():
        raw_name = str(row["Name"])
        norm_name = _normalize_name(raw_name)
        # Also expand via alias before tokenizing
        expanded = _ALIASES.get(norm_name, norm_name)
        tokens = frozenset(expanded.split()) - _STOP_WORDS
        rank = int(row["Rank"])
        employees = int(row["_employees_int"])
        industry = str(row.get("Industry", "")).strip()
        lookup.append({
            "tokens": tokens,
            "rank": rank,
            "employees": employees,
            "name": raw_name,
            "industry": industry,
        })

    return lookup, max_log_emp


# ──────────────────────────────────────────────────────────────────────────────
# Core matching function
# ──────────────────────────────────────────────────────────────────────────────

def _match_company(name: str, lookup: list, threshold: float = MATCH_THRESHOLD):
    """
    Try to match a single company name against the Fortune 100 lookup.

    Returns (matched_row_dict | None, score)
    """
    if _is_placeholder(name):
        return None, 0.0

    query_tokens = _tokenize(name)
    if not query_tokens:
        return None, 0.0

    best_score = 0.0
    best_row = None
    for entry in lookup:
        ref_tokens = entry["tokens"]
        score = _token_set_score(query_tokens, ref_tokens)
        if score > best_score:
            best_score = score
            best_row = entry

    if best_score >= threshold:
        return best_row, best_score
    return None, best_score


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────

def compute_company_features(company_list_str, csv_path: str) -> dict:
    """
    Given a raw `professional_company_names` value (Python-list string) and the
    path to Largest_Companies.csv, return a dict with three features.

    Parameters
    ----------
    company_list_str : str | float
        Raw value from the CSV, e.g. "['Meta Inc.', 'Google']"
    csv_path : str
        Absolute path to Largest_Companies.csv

    Returns
    -------
    dict with keys:
        company_is_fortune100      : int   (0 or 1)
        company_fortune100_rank_norm: float (0.0 – 1.0)
        company_size_norm          : float (0.0 – 1.0)
    """
    lookup, max_log_emp = _load_fortune_lookup(csv_path)

    companies = _safe_parse_list(company_list_str)

    best_rank  = None
    best_emp   = 0

    for company in companies:
        row, score = _match_company(company, lookup)
        if row is None:
            continue
        rank = row["rank"]
        emp  = row["employees"]
        # Track best (lowest numbered) rank = highest prestige
        if best_rank is None or rank < best_rank:
            best_rank = rank
        if emp > best_emp:
            best_emp = emp

    if best_rank is None:
        return {
            "company_is_fortune100":       0,
            "company_fortune100_rank_norm": 0.0,
            "company_size_norm":           0.0,
        }

    # Rank normalized: rank 1 → 1.0, rank 100 → 1 - 99/99 = 0.0
    rank_norm = 1.0 - (best_rank - 1) / 99.0

    # Employee count normalized (log10 scale)
    size_norm = math.log10(max(best_emp, 1)) / max_log_emp if max_log_emp > 0 else 0.0
    size_norm = min(size_norm, 1.0)

    return {
        "company_is_fortune100":       1,
        "company_fortune100_rank_norm": round(rank_norm, 4),
        "company_size_norm":           round(size_norm, 4),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Coverage analysis helper
# ──────────────────────────────────────────────────────────────────────────────

def analyse_coverage(data_path: str, csv_path: str) -> None:
    """Print a coverage report: how many rows match at least one Fortune 100 firm."""
    df = pd.read_csv(data_path)
    lookup, max_log_emp = _load_fortune_lookup(csv_path)

    matched = 0
    matched_companies = {}
    for val in df["professional_company_names"]:
        companies = _safe_parse_list(val)
        row_matched = False
        for company in companies:
            row, score = _match_company(company, lookup)
            if row is not None:
                row_matched = True
                name = row["name"]
                matched_companies[name] = matched_companies.get(name, 0) + 1
        if row_matched:
            matched += 1

    total = len(df)
    print(f"\nCoverage: {matched}/{total} rows ({matched/total*100:.1f}%) matched ≥1 Fortune 100 company")
    print("\nFortune 100 companies found in resume data:")
    for name, count in sorted(matched_companies.items(), key=lambda x: -x[1]):
        print(f"  {count:4d}  {name}")


# ──────────────────────────────────────────────────────────────────────────────
# CLI quick test
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import os
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path  = os.path.join(base, "data", "resume_data.csv")
    csv_path   = os.path.join(base, "data", "Largest_Companies.csv")

    print("=== Unit tests ===")
    test_cases = [
        ("['Meta Inc.']",            True),
        ("['Google']",               True),   # alias → Alphabet
        ("['Microsoft']",            True),
        ("['Tech Mahindra']",        False),
        ("['Company Name']",         False),
        ("['N/A', 'Company Name']",  False),
        ("['Amazon', 'Accenture']",  True),
        ("['General Electric']",     True),
        ("['Daffodil Software Pvt Ltd']", False),
    ]
    for val, expected in test_cases:
        result = compute_company_features(val, csv_path)
        matched = bool(result["company_is_fortune100"])
        status = "OK" if matched == expected else "FAIL"
        print(f"  [{status}]  {val:40s}  is_f100={matched}  rank_norm={result['company_fortune100_rank_norm']:.3f}  size={result['company_size_norm']:.3f}")

    print()
    analyse_coverage(data_path, csv_path)
