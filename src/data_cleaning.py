"""
data_cleaning.py
----------------
Reads data/resume_data.csv, cleans and transforms every column, and writes
data/cleaned_resume_data.csv.

Design decisions (confirmed with team):
  - Task type      : REGRESSION (predict continuous matched_score 0-1)
  - Text embeddings: SBERT (long text) + Word2Vec (skill lists) — done at model time
  - Missing text   : zero-vector + 'has_career_objective' binary flag
  - Duplicates     : responsibilities.1 is 100% identical to responsibilities → dropped

Column fate summary
-------------------
DROPPED (useless / >79% missing / pure URL):
    address, company_urls, online_links, extra_curricular_organization_links,
    responsibilities.1, extra_curricular_organization_names,
    educational_institution_name, certification_providers, issue_dates, expiry_dates,
    languages, proficiency_levels, extra_curricular_activity_types,
    extra_curricular_organization_links, role_positions, online_links
    NOTE: professional_company_names is used to extract Fortune 100 features BEFORE dropping

NUMERICAL (parsed from messy strings):
    experiencere_requirement  → experience_min_years
    age_requirement           → age_min, age_max
    passing_years             → years_since_graduation
    educational_results       → gpa_normalized  (0-1 scale)
    start_dates + end_dates   → total_work_experience_years

COMPANY FEATURES (matched against Fortune 100 via Largest_Companies.csv):
    professional_company_names → company_is_fortune100 (binary)
                               → company_fortune100_rank_norm (0-1, 1=rank#1/Walmart)
                               → company_size_norm (0-1, log-normalized employee count)

UNIVERSITY RANKING FEATURES (matched against CWUR/Shanghai/Times rankings):
    educational_institution_name → university_rank_score (0-1)
                                 → university_world_rank (0-1, 1=rank#1)
                                 → university_is_ranked (binary)

BINARY FLAGS:
    career_objective          → has_career_objective
    certification_skills      → has_certification, num_certifications

TEXT (cleaned, stored as-is for later embedding):
    career_objective, skills, responsibilities, skills_required,
    educationaL_requirements, related_skils_in_job, major_field_of_studies

CATEGORICAL (normalized strings, one-hot at model time):
    job_position_name, degree_level, result_type, positions (most-recent title)

TARGET:
    matched_score  (unchanged)
"""

import re
import ast
import os
import pandas as pd
import numpy as np
from datetime import datetime

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

CURRENT_YEAR = 2025  # use a fixed reference year for reproducibility


def _safe_parse_list(val):
    """Parse a Python-list string like "['A', 'B']" into a Python list."""
    if pd.isna(val) or str(val).strip() == "":
        return []
    try:
        result = ast.literal_eval(str(val))
        if isinstance(result, list):
            return [str(x).strip() for x in result]
        return [str(result).strip()]
    except Exception:
        # fallback: split on comma
        return [x.strip().strip("'\"[]") for x in str(val).split(",") if x.strip()]


def _clean_text(val):
    """Lower-case, collapse whitespace, strip newlines."""
    if pd.isna(val):
        return ""
    s = str(val)
    s = s.replace("\n", " ").replace("\r", " ")
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def _parse_experience_years(val):
    """
    Convert experience strings to a minimum-years integer.
    Examples:
      'At least 3 years'  → 3
      '2 to 5 years'      → 2
      'At least 1 year'   → 1
      NaN                 → 0
    """
    if pd.isna(val):
        return 0.0
    s = str(val).lower()
    # 'X to Y years' → X
    m = re.search(r"(\d+)\s*to\s*(\d+)", s)
    if m:
        return float(m.group(1))
    # 'at least X year(s)'
    m = re.search(r"(\d+)", s)
    if m:
        return float(m.group(1))
    return 0.0


def _parse_age_requirement(val):
    """
    Return (age_min, age_max) from strings like:
      'Age 22 to 30 years'    → (22, 30)
      'Age at least 24 years' → (24, 65)
      'Age at most 40 years'  → (18, 40)
      NaN                     → (0, 0)
    """
    if pd.isna(val):
        return 0.0, 0.0
    s = str(val).lower()
    m = re.search(r"(\d+)\s*to\s*(\d+)", s)
    if m:
        return float(m.group(1)), float(m.group(2))
    m_least = re.search(r"at least\s*(\d+)", s)
    if m_least:
        return float(m_least.group(1)), 65.0
    m_most = re.search(r"at most\s*(\d+)", s)
    if m_most:
        return 18.0, float(m_most.group(1))
    # fallback: first number found
    m = re.search(r"(\d+)", s)
    if m:
        return float(m.group(1)), 65.0
    return 0.0, 0.0


def _parse_years_since_graduation(val):
    """
    Extract most recent graduation year from a list-string, return
    CURRENT_YEAR - year.  Returns 0 on failure.
    """
    years = _safe_parse_list(val)
    parsed = []
    for y in years:
        m = re.search(r"(19|20)\d{2}", str(y))
        if m:
            parsed.append(int(m.group()))
    if not parsed:
        return 0.0
    return float(CURRENT_YEAR - max(parsed))


def _parse_gpa(val):
    """
    Normalise educational_results to 0-1.
    Handles:
      '3.5 out of 4.0'  → 3.5/4.0 = 0.875
      '85%'             → 0.85
      '3.5'             → heuristic (>4.0 → /100, else /4.0 or /5.0)
      NaN               → 0.5 (neutral imputation)
    """
    if pd.isna(val):
        return 0.5
    # Attempt to parse list first
    items = _safe_parse_list(val)
    scores = []
    for item in items:
        s = str(item).lower().strip()
        # 'X out of Y'
        m = re.search(r"([\d.]+)\s*out\s*of\s*([\d.]+)", s)
        if m:
            num, denom = float(m.group(1)), float(m.group(2))
            if denom > 0:
                scores.append(min(num / denom, 1.0))
            continue
        # percentage
        m = re.search(r"([\d.]+)\s*%", s)
        if m:
            scores.append(min(float(m.group(1)) / 100.0, 1.0))
            continue
        # bare number (must start and end with digit to avoid lone '.')
        m = re.search(r"\d[\d.]*\d|\d", s)
        if m:
            v = float(m.group(0))
            if v > 10:      # likely a percentage
                scores.append(min(v / 100.0, 1.0))
            elif v <= 4.0:  # GPA out of 4
                scores.append(v / 4.0)
            elif v <= 5.0:  # GPA out of 5
                scores.append(v / 5.0)
            else:
                scores.append(min(v / 100.0, 1.0))
    if not scores:
        return 0.5
    return float(np.mean(scores))


def _parse_work_experience(start_col, end_col):
    """
    Given two list-string columns, compute total years of work experience.
    Dates can be 'YYYY', 'Month YYYY', 'YYYY-MM', etc.
    """

    def _extract_year(s):
        m = re.search(r"(19|20)\d{2}", str(s))
        return int(m.group()) if m else None

    def _row_experience(start_val, end_val):
        starts = _safe_parse_list(start_val)
        ends = _safe_parse_list(end_val)
        total = 0.0
        for s, e in zip(starts, ends):
            sy = _extract_year(s)
            if sy is None:
                continue
            ey_str = str(e).lower().strip()
            if "present" in ey_str or "current" in ey_str or ey_str == "":
                ey = CURRENT_YEAR
            else:
                ey = _extract_year(e) or CURRENT_YEAR
            total += max(0.0, ey - sy)
        return total

    return start_col, end_col, _row_experience


def _normalize_degree(val):
    """Map degree list to highest-level degree label."""
    if pd.isna(val):
        return "Unknown"
    items = _safe_parse_list(val)
    text = " ".join(items).lower()
    if "phd" in text or "doctor" in text or "ph.d" in text:
        return "PhD"
    if ("master" in text or "m.sc" in text or "msc" in text or "mba" in text
            or "m.eng" in text or re.search(r"\bm\.s\b", text) or "m.s." in text
            or re.search(r"\bms\b", text)):
        return "Masters"
    if ("bachelor" in text or "b.sc" in text or "bsc" in text or "b.tech" in text
            or "b.eng" in text or "b.a." in text or re.search(r"\bb\.s\b", text)
            or "b.s." in text or "bca" in text or "bba" in text or "b.com" in text
            or re.search(r"\bbs\b", text) or "honours" in text or "honor" in text):
        return "Bachelors"
    if "diploma" in text or "associate" in text or "certificate" in text:
        return "Diploma/Certificate"
    if items:
        return "Other"
    return "Unknown"


def _normalize_result_type(val):
    """Map result_type to one of: GPA, CGPA, Percentage, Grade, Other."""
    if pd.isna(val):
        return "Other"
    items = _safe_parse_list(val)
    text = " ".join(items).lower()
    if "cgpa" in text:
        return "CGPA"
    if "gpa" in text:
        return "GPA"
    if "percent" in text or "%" in text:
        return "Percentage"
    if "grade" in text or "division" in text:
        return "Grade"
    return "Other"


def _normalize_job_position(val):
    """Lower-case, strip, collapse whitespace."""
    if pd.isna(val):
        return "unknown"
    return re.sub(r"\s+", " ", str(val).lower().strip())


def _most_recent_position(val):
    """Return the first (most recent) position from a list-string."""
    items = _safe_parse_list(val)
    if items:
        return items[0][:60].lower().strip()
    return "unknown"


def _skills_to_text(val):
    """Parse skill list into space-separated lowercase string."""
    items = _safe_parse_list(val)
    return " ".join(x.lower() for x in items if x)


def _newline_text_to_clean(val):
    """Convert newline-delimited items to space-separated clean text."""
    return _clean_text(val)


# ──────────────────────────────────────────────────────────────────────────────
# Main cleaning function
# ──────────────────────────────────────────────────────────────────────────────

def clean_resume_data(input_path: str, output_path: str,
                      ranking_dir: str = None,
                      companies_csv: str = None) -> pd.DataFrame:
    print(f"Loading {input_path} ...")
    df = pd.read_csv(input_path)
    print(f"  Raw shape: {df.shape}")

    # Ensure src/ directory is on path for sibling module imports
    import sys
    _src = os.path.dirname(os.path.abspath(__file__))
    if _src not in sys.path:
        sys.path.insert(0, _src)

    # ── 0. University ranking features (uses educational_institution_name) ──
    if ranking_dir and os.path.isdir(ranking_dir):
        print("  Computing university ranking features ...")
        from university_ranking import compute_university_rank_features
        rank_features = df["educational_institution_name"].apply(
            lambda v: compute_university_rank_features(str(v) if not pd.isna(v) else "", ranking_dir)
        )
        df["university_rank_score"] = rank_features.apply(lambda x: x["university_rank_score"])
        df["university_world_rank"] = rank_features.apply(lambda x: x["university_world_rank"])
        df["university_is_ranked"]  = rank_features.apply(lambda x: x["university_is_ranked"])
        n_ranked = df["university_is_ranked"].sum()
        print(f"    Ranked institutions found: {n_ranked}/{len(df)} rows ({n_ranked/len(df)*100:.1f}%)")
    else:
        print("  Skipping university rankings (no ranking_dir provided or not found).")
        df["university_rank_score"] = 0.0
        df["university_world_rank"] = 0.0
        df["university_is_ranked"]  = 0

    # ── 0b. Fortune 100 company features (uses professional_company_names) ──
    if companies_csv and os.path.isfile(companies_csv):
        print("  Computing Fortune 100 company features ...")
        from company_features import compute_company_features
        comp_features = df["professional_company_names"].apply(
            lambda v: compute_company_features(str(v) if not pd.isna(v) else "", companies_csv)
        )
        df["company_is_fortune100"]        = comp_features.apply(lambda x: x["company_is_fortune100"])
        df["company_fortune100_rank_norm"] = comp_features.apply(lambda x: x["company_fortune100_rank_norm"])
        df["company_size_norm"]            = comp_features.apply(lambda x: x["company_size_norm"])
        n_matched = df["company_is_fortune100"].sum()
        print(f"    Fortune 100 matches: {n_matched}/{len(df)} rows ({n_matched/len(df)*100:.1f}%)")
    else:
        print("  Skipping Fortune 100 company features (no companies_csv provided or not found).")
        df["company_is_fortune100"]        = 0
        df["company_fortune100_rank_norm"] = 0.0
        df["company_size_norm"]            = 0.0

    # ── 1. Drop useless columns ─────────────────────────────────────────────
    drop_cols = [
        "address",
        "company_urls",
        "online_links",
        "extra_curricular_organization_links",
        "responsibilities.1",          # 100% duplicate of responsibilities
        "professional_company_names",  # high-cardinality names
        "extra_curricular_organization_names",
        "educational_institution_name",   # used above for ranking features; now drop
        "certification_providers",
        "issue_dates",
        "expiry_dates",
        "languages",
        "proficiency_levels",
        "extra_curricular_activity_types",
        "role_positions",
    ]
    df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True)

    # ── 2. Numerical features ────────────────────────────────────────────────
    print("  Parsing numerical features ...")

    # experience
    df["experience_min_years"] = df["experiencere_requirement"].apply(_parse_experience_years)
    df.drop(columns=["experiencere_requirement"], inplace=True)

    # age
    age_parsed = df["age_requirement"].apply(_parse_age_requirement)
    df["age_min"] = age_parsed.apply(lambda x: x[0])
    df["age_max"] = age_parsed.apply(lambda x: x[1])
    df.drop(columns=["age_requirement"], inplace=True)

    # years since graduation
    df["years_since_graduation"] = df["passing_years"].apply(_parse_years_since_graduation)
    df.drop(columns=["passing_years"], inplace=True)

    # GPA normalized
    df["gpa_normalized"] = df["educational_results"].apply(_parse_gpa)
    df.drop(columns=["educational_results"], inplace=True)

    # total work experience
    row_exp_fn = _parse_work_experience(None, None)[2]
    df["total_work_experience_years"] = df.apply(
        lambda r: row_exp_fn(r["start_dates"], r["end_dates"]), axis=1
    )
    df.drop(columns=["start_dates", "end_dates"], inplace=True)

    # certifications binary flag
    df["has_certification"] = (~df["certification_skills"].isna()).astype(int)
    df["num_certifications"] = df["certification_skills"].apply(
        lambda v: len(_safe_parse_list(v)) if not pd.isna(v) else 0
    )
    df.drop(columns=["certification_skills"], inplace=True)

    # ── 3. Binary flags ──────────────────────────────────────────────────────
    df["has_career_objective"] = (~df["career_objective"].isna() & (df["career_objective"].str.strip() != "")).astype(int)

    # ── 4. Text columns — clean and keep as text ─────────────────────────────
    print("  Cleaning text columns ...")
    df["career_objective"]       = df["career_objective"].apply(_clean_text)
    df["skills"]                 = df["skills"].apply(_skills_to_text)
    df["responsibilities"]       = df["responsibilities"].apply(_newline_text_to_clean)
    df["skills_required"]        = df["skills_required"].apply(_newline_text_to_clean)
    df["educationaL_requirements"] = df["educationaL_requirements"].apply(_clean_text)
    df["related_skils_in_job"]   = df["related_skils_in_job"].apply(_clean_text)
    df["major_field_of_studies"] = df["major_field_of_studies"].apply(
        lambda v: " ".join(_safe_parse_list(v)).lower()
    )

    # ── 5. Categorical columns ───────────────────────────────────────────────
    print("  Normalising categorical columns ...")
    df["job_position_name"] = df["job_position_name"].apply(_normalize_job_position)
    df["degree_level"]      = df["degree_names"].apply(_normalize_degree)
    df["result_type"]       = df["result_types"].apply(_normalize_result_type)
    df["most_recent_position"] = df["positions"].apply(_most_recent_position)
    df.drop(columns=["degree_names", "result_types", "positions"], inplace=True)

    # ── 6. Remaining columns with low info — drop ────────────────────────────
    more_drop = ["locations", "extra_curricular_organization_links"]
    df.drop(columns=[c for c in more_drop if c in df.columns], inplace=True)

    # ── 7. Final clean-up ────────────────────────────────────────────────────
    # Clip experience / age to sensible ranges
    df["experience_min_years"]       = df["experience_min_years"].clip(0, 30)
    df["age_min"]                    = df["age_min"].clip(0, 70)
    df["age_max"]                    = df["age_max"].clip(0, 70)
    df["years_since_graduation"]     = df["years_since_graduation"].clip(0, 50)
    df["total_work_experience_years"] = df["total_work_experience_years"].clip(0, 50)
    df["gpa_normalized"]             = df["gpa_normalized"].clip(0.0, 1.0)
    df["university_rank_score"]        = df["university_rank_score"].clip(0.0, 1.0)
    df["university_world_rank"]        = df["university_world_rank"].clip(0.0, 1.0)
    df["company_fortune100_rank_norm"] = df["company_fortune100_rank_norm"].clip(0.0, 1.0)
    df["company_size_norm"]            = df["company_size_norm"].clip(0.0, 1.0)

    # Drop any remaining unnamed / index columns
    df = df.loc[:, ~df.columns.str.startswith("Unnamed")]

    # Reset index
    df.reset_index(drop=True, inplace=True)

    print(f"  Cleaned shape: {df.shape}")
    print(f"  Columns: {list(df.columns)}")
    print(f"\n  Null counts after cleaning:")
    print(df.isnull().sum()[df.isnull().sum() > 0])

    df.to_csv(output_path, index=False)
    print(f"\nSaved to {output_path}")
    return df


# ──────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    clean_resume_data(
        input_path=os.path.join(base, "data", "resume_data.csv"),
        output_path=os.path.join(base, "data", "cleaned_resume_data.csv"),
        ranking_dir=os.path.join(base, "data", "university_rankings"),
        companies_csv=os.path.join(base, "data", "Largest_Companies.csv"),
    )
