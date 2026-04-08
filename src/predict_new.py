"""
predict_new.py
--------------
Load hand-crafted new-data samples (resume + job pairs) and run them through
the trained SBERT ResumeMatchNet to produce match-score predictions.

Usage:
    python3 src/predict_new.py

Reads  : data/new_candidate_data.csv   (built by this script's build_csv())
Writes : prints a prediction table to stdout
"""

import os, sys, pickle
import numpy as np
import pandas as pd
import torch

_SRC = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_SRC)
sys.path.insert(0, _SRC)

from dataset import (
    NUMERICAL_COLS, SBERT_TEXT_COLS, W2V_TEXT_COLS, SBERT_DIM, W2V_DIM,
    _encode_categoricals, _compute_sbert_embeddings, _text_to_w2v_vector,
)
from model import ResumeMatchNet


# ══════════════════════════════════════════════════════════════════════════════
# 1.  Build the new-data CSV
# ══════════════════════════════════════════════════════════════════════════════

# ── Job descriptions (3 roles) ────────────────────────────────────────────────

JOBS = {
    "mne summer development analyst": {
        "responsibilities": (
            "Presenting presentations and giving market news updates. "
            "Participating in intense and involved projects. "
            "Attending networking sessions and tailored learning programs."
        ),
        "educationaL_requirements": (
            "First year or later at the University of Toronto."
        ),
        "skills_required": (
            "Excel analytical skills presentations market news"
        ),
        "related_skils_in_job": "Excel analytical skills financial markets",
        "experience_min_years": 0.0,
        "age_min": 0.0,
        "age_max": 0.0,
    },
    "mne senior analyst": {
        "responsibilities": (
            "Leading members and editing reports. "
            "Presenting presentations and giving market news updates. "
            "Organizing and managing SGC charity events. "
            "Developing leadership skills through hands-on experience."
        ),
        "educationaL_requirements": (
            "Ideally one year of experience in St. George Capital. "
            "Should be at the University of Toronto. "
            "Past leadership experience required."
        ),
        "skills_required": (
            "Excel analytical skills leadership report editing presentations market news"
        ),
        "related_skils_in_job": "Excel leadership analytical skills financial markets",
        "experience_min_years": 1.0,
        "age_min": 0.0,
        "age_max": 0.0,
    },
    "mne analyst - industrials": {
        "responsibilities": (
            "Attending meetings and becoming an expert in industrial companies. "
            "Conducting research and analysis on industrial sector firms. "
            "Presenting findings to the team."
        ),
        "educationaL_requirements": (
            "Second year or later at the University of Toronto. "
            "Ideally in Industrial Engineering."
        ),
        "skills_required": (
            "Excel analytical skills financial markets industrial sector knowledge"
        ),
        "related_skils_in_job": "Excel analytical skills financial markets industrials",
        "experience_min_years": 0.0,
        "age_min": 0.0,
        "age_max": 0.0,
    },
}

# ── Candidate data (extracted from 24 resume PDFs) ───────────────────────────

CANDIDATES = [
    # (name, assigned_job_key, resume_fields_dict)
    ("Henley He", "mne summer development analyst", {
        "career_objective": "",
        "skills": "python c java latex matlab autocad adobe express adobe premiere pro adobe photoshop arduino canva onshape microsoft power bi microsoft visio process map making flow chart making",
        "major_field_of_studies": "engineering science math stats finance",
        "gpa_normalized": 0.0,
        "total_work_experience_years": 2.2,
        "has_certification": 1,
        "num_certifications": 2,
        "has_career_objective": 0,
        "degree_level": "Bachelors",
        "result_type": "Other",
        "most_recent_position": "operations/logistics coordinator",
        "years_since_graduation": 0.0,
        "university": "University of Toronto",
        "companies": "",
    }),
    ("Rita Huang", "mne summer development analyst", {
        "career_objective": "",
        "skills": "java python r microsoft office suite adobe photoshop davinci resolve organization problem solving leadership communication collaboration attention to detail",
        "major_field_of_studies": "mathematics statistics east asian studies",
        "gpa_normalized": 0.7875,
        "total_work_experience_years": 1.4,
        "has_certification": 1,
        "num_certifications": 1,
        "has_career_objective": 0,
        "degree_level": "Bachelors",
        "result_type": "GPA",
        "most_recent_position": "freelance tutor",
        "years_since_graduation": 0.0,
        "university": "University of Toronto",
        "companies": "",
    }),
    ("Arman Zaher", "mne senior analyst", {
        "career_objective": "",
        "skills": "python java sql matlab r c c++ ampl scikit-learn pytorch pandas langchain neo4j matplotlib gurobi simpy plotly dash mlflow docker linux azure excel visio ms project power bi power apps power automate gcp bigquery geopandas fastapi",
        "major_field_of_studies": "industrial systems engineering artificial intelligence business",
        "gpa_normalized": 0.0,
        "total_work_experience_years": 2.15,
        "has_certification": 0,
        "num_certifications": 0,
        "has_career_objective": 0,
        "degree_level": "Bachelors",
        "result_type": "Other",
        "most_recent_position": "data scientist intern",
        "years_since_graduation": 0.0,
        "university": "University of Toronto",
        "companies": "Royal Bank of Canada Bombardier",
    }),
    ("Katie Ma", "mne senior analyst", {
        "career_objective": "",
        "skills": "python engineering design carbon fibre fabrication cpr standard first aid leadership competitive swimming",
        "major_field_of_studies": "engineering science",
        "gpa_normalized": 1.0,
        "total_work_experience_years": 1.1,
        "has_certification": 0,
        "num_certifications": 0,
        "has_career_objective": 0,
        "degree_level": "Bachelors",
        "result_type": "GPA",
        "most_recent_position": "swim instructor and lifeguard",
        "years_since_graduation": 0.0,
        "university": "University of Toronto",
        "companies": "",
    }),
    ("Sophie Shen", "mne senior analyst", {
        "career_objective": "",
        "skills": "bloomberg terminal factset capitaliq stata r python ms office equity research market research teaching",
        "major_field_of_studies": "commerce finance economics",
        "gpa_normalized": 1.0,
        "total_work_experience_years": 3.25,
        "has_certification": 0,
        "num_certifications": 0,
        "has_career_objective": 0,
        "degree_level": "Bachelors",
        "result_type": "GPA",
        "most_recent_position": "strategy and marketing analyst",
        "years_since_graduation": 0.0,
        "university": "University of Toronto",
        "companies": "",
    }),
    ("Sherry Hong", "mne senior analyst", {
        "career_objective": "",
        "skills": "microsoft office r python excel stata financial analysis equity research market research data cleaning data visualization",
        "major_field_of_studies": "statistical science economics",
        "gpa_normalized": 0.955,
        "total_work_experience_years": 0.59,
        "has_certification": 0,
        "num_certifications": 0,
        "has_career_objective": 0,
        "degree_level": "Bachelors",
        "result_type": "GPA",
        "most_recent_position": "product manager intern",
        "years_since_graduation": 0.0,
        "university": "University of Toronto",
        "companies": "",
    }),
    ("Lucy Yoon", "mne analyst - industrials", {
        "career_objective": "",
        "skills": "python microsoft excel pivot tables vlookup powerpoint canva data analysis social media marketing",
        "major_field_of_studies": "commerce",
        "gpa_normalized": 0.0,
        "total_work_experience_years": 0.5,
        "has_certification": 0,
        "num_certifications": 0,
        "has_career_objective": 0,
        "degree_level": "Bachelors",
        "result_type": "Other",
        "most_recent_position": "operations manager",
        "years_since_graduation": 0.0,
        "university": "University of Toronto",
        "companies": "",
    }),
    ("Mohamad Majzoub", "mne analyst - industrials", {
        "career_objective": "",
        "skills": "python excel surpac matlab r json french english arabic",
        "major_field_of_studies": "mineral mining engineering",
        "gpa_normalized": 0.0,
        "total_work_experience_years": 2.0,
        "has_certification": 0,
        "num_certifications": 0,
        "has_career_objective": 0,
        "degree_level": "Bachelors",
        "result_type": "Other",
        "most_recent_position": "co-owner",
        "years_since_graduation": 0.0,
        "university": "University of Toronto",
        "companies": "",
    }),
    ("Marvin Chen", "mne analyst - industrials", {
        "career_objective": "",
        "skills": "microsoft office excel power bi ms project python matlab",
        "major_field_of_studies": "chemical engineering",
        "gpa_normalized": 0.925,
        "total_work_experience_years": 0.17,
        "has_certification": 0,
        "num_certifications": 0,
        "has_career_objective": 0,
        "degree_level": "Bachelors",
        "result_type": "GPA",
        "most_recent_position": "production intern",
        "years_since_graduation": 0.0,
        "university": "University of Toronto",
        "companies": "BASF",
    }),
    ("Amy Xu", "mne analyst - industrials", {
        "career_objective": "",
        "skills": "python c matlab calculus numerical methods linear algebra excel vernier logger pro mathematical modelling experimental design data analysis",
        "major_field_of_studies": "engineering science",
        "gpa_normalized": 0.0,
        "total_work_experience_years": 2.3,
        "has_certification": 0,
        "num_certifications": 0,
        "has_career_objective": 0,
        "degree_level": "Bachelors",
        "result_type": "Other",
        "most_recent_position": "mathematics tutor",
        "years_since_graduation": 0.0,
        "university": "University of Toronto",
        "companies": "",
    }),
    ("Dylan Luong", "mne analyst - industrials", {
        "career_objective": "",
        "skills": "advanced excel canva financial modelling github python sql wix",
        "major_field_of_studies": "economics data analytics",
        "gpa_normalized": 0.8725,
        "total_work_experience_years": 1.67,
        "has_certification": 1,
        "num_certifications": 2,
        "has_career_objective": 0,
        "degree_level": "Bachelors",
        "result_type": "GPA",
        "most_recent_position": "speaker series executive",
        "years_since_graduation": 0.0,
        "university": "University of Toronto",
        "companies": "",
    }),
    ("Sara Salguero", "mne analyst - industrials", {
        "career_objective": "",
        "skills": "excel powerpoint basic python crm systems graphic design financial analysis cold outreach sponsorship strategy",
        "major_field_of_studies": "commerce",
        "gpa_normalized": 1.0,
        "total_work_experience_years": 3.0,
        "has_certification": 1,
        "num_certifications": 3,
        "has_career_objective": 0,
        "degree_level": "Bachelors",
        "result_type": "GPA",
        "most_recent_position": "intern",
        "years_since_graduation": 0.0,
        "university": "University of Toronto",
        "companies": "",
    }),
    ("Kian Keshtkar", "mne analyst - industrials", {
        "career_objective": "",
        "skills": "equity research market trend analysis financial modeling excel pivot tables power bi tableau python pandas numpy sql r statistical analysis data visualization exploratory analysis microsoft office suite jupyter machine learning",
        "major_field_of_studies": "applied statistics computer science",
        "gpa_normalized": 0.0,
        "total_work_experience_years": 1.9,
        "has_certification": 0,
        "num_certifications": 0,
        "has_career_objective": 0,
        "degree_level": "Bachelors",
        "result_type": "Other",
        "most_recent_position": "business analyst and developer",
        "years_since_graduation": 0.0,
        "university": "University of Toronto",
        "companies": "",
    }),
    ("Yenilmez Oguz Silahtaroglu", "mne analyst - industrials", {
        "career_objective": "",
        "skills": "python java sql excel powerpoint bloomberg terminal english german turkish",
        "major_field_of_studies": "mathematics economics computer science",
        "gpa_normalized": 0.9325,
        "total_work_experience_years": 0.25,
        "has_certification": 0,
        "num_certifications": 0,
        "has_career_objective": 0,
        "degree_level": "Bachelors",
        "result_type": "GPA",
        "most_recent_position": "strategy intern",
        "years_since_graduation": 0.0,
        "university": "University of Toronto",
        "companies": "",
    }),
    ("Juan Pablo Escobedo del Villar", "mne analyst - industrials", {
        "career_objective": "",
        "skills": "excel financial accounting financial markets management calculus",
        "major_field_of_studies": "commerce",
        "gpa_normalized": 0.0,
        "total_work_experience_years": 0.0,
        "has_certification": 0,
        "num_certifications": 0,
        "has_career_objective": 0,
        "degree_level": "Bachelors",
        "result_type": "Other",
        "most_recent_position": "donations coordinator",
        "years_since_graduation": 0.0,
        "university": "University of Toronto",
        "companies": "",
    }),
    ("Lincoln Cheng", "mne analyst - industrials", {
        "career_objective": "",
        "skills": "python numpy matplotlib yfinance jupyter notebook sql c# matlab bloomberg terminal java openai api mandarin cantonese",
        "major_field_of_studies": "mathematics computer science economics",
        "gpa_normalized": 0.9725,
        "total_work_experience_years": 0.67,
        "has_certification": 0,
        "num_certifications": 0,
        "has_career_objective": 0,
        "degree_level": "Bachelors",
        "result_type": "GPA",
        "most_recent_position": "wealth management intern",
        "years_since_graduation": 0.0,
        "university": "University of Toronto",
        "companies": "",
    }),
    ("Ying Kit Arther Wan", "mne analyst - industrials", {
        "career_objective": "Aspiring finance and economics specialist with a statistics major. Fluent in Cantonese Mandarin and English. Proficient with Python Java R Studio Revit. Participated in 7+ case competitions. 200+ hours of community service.",
        "skills": "python java r studio revit 3d modeling leadership cantonese mandarin english",
        "major_field_of_studies": "finance economics statistics",
        "gpa_normalized": 0.85,
        "total_work_experience_years": 0.5,
        "has_certification": 0,
        "num_certifications": 0,
        "has_career_objective": 1,
        "degree_level": "Bachelors",
        "result_type": "GPA",
        "most_recent_position": "equity research analyst",
        "years_since_graduation": 0.0,
        "university": "University of Toronto",
        "companies": "",
    }),
    ("Emanuel Abdelmalak", "mne analyst - industrials", {
        "career_objective": "",
        "skills": "financial modeling data analysis probability statistics python r risk assessment budgeting forecasting project management excel access sql executive presentations data reporting ai microsoft office suite",
        "major_field_of_studies": "actuarial science data analytics",
        "gpa_normalized": 0.0,
        "total_work_experience_years": 1.4,
        "has_certification": 0,
        "num_certifications": 0,
        "has_career_objective": 0,
        "degree_level": "Bachelors",
        "result_type": "Other",
        "most_recent_position": "machine learning research fellow",
        "years_since_graduation": 0.0,
        "university": "University of Toronto",
        "companies": "",
    }),
    ("Vanessa Huo", "mne analyst - industrials", {
        "career_objective": "",
        "skills": "python java c c++ matlab r pytorch tensorflow scikit-learn numpy pandas opencv yolo typescript javascript html css react next.js fastapi sql azure mongodb deep learning cnns rnns vaes transformers fpga verilog",
        "major_field_of_studies": "engineering science machine intelligence",
        "gpa_normalized": 0.0,
        "total_work_experience_years": 0.33,
        "has_certification": 0,
        "num_certifications": 0,
        "has_career_objective": 0,
        "degree_level": "Bachelors",
        "result_type": "Other",
        "most_recent_position": "co-operative education student",
        "years_since_graduation": 0.0,
        "university": "University of Toronto",
        "companies": "",
    }),
    ("Sheraz Arshad", "mne analyst - industrials", {
        "career_objective": "",
        "skills": "python sql haskell c c++ excel powerpoint fastapi pandas numpy opencv docker google vertex ai gemini api aws postgresql rest apis git microcontrollers",
        "major_field_of_studies": "mathematics statistics physics",
        "gpa_normalized": 0.0,
        "total_work_experience_years": 0.83,
        "has_certification": 0,
        "num_certifications": 0,
        "has_career_objective": 0,
        "degree_level": "Bachelors",
        "result_type": "Other",
        "most_recent_position": "backend engineering lead",
        "years_since_graduation": 0.0,
        "university": "McMaster University",
        "companies": "",
    }),
    ("Joanna Tai", "mne analyst - industrials", {
        "career_objective": "",
        "skills": "word microsoft excel powerpoint canva video editing communication teamwork problem-solving adaptability",
        "major_field_of_studies": "economics industrial relations human resources",
        "gpa_normalized": 0.0,
        "total_work_experience_years": 0.0,
        "has_certification": 0,
        "num_certifications": 0,
        "has_career_objective": 0,
        "degree_level": "Bachelors",
        "result_type": "Other",
        "most_recent_position": "co-founder and director of finance",
        "years_since_graduation": 0.0,
        "university": "University of Toronto",
        "companies": "",
    }),
    ("Andrew Wu", "mne analyst - industrials", {
        "career_objective": "",
        "skills": "python pandas numpy scikit-learn sql excel data analysis pivot tables powerbi statistical modeling logistic regression random forest probability statistics quantitative trading signal development market data analysis strategy research risk-based decision modeling",
        "major_field_of_studies": "mathematical applications economics finance statistics",
        "gpa_normalized": 0.95,
        "total_work_experience_years": 0.5,
        "has_certification": 0,
        "num_certifications": 0,
        "has_career_objective": 0,
        "degree_level": "Bachelors",
        "result_type": "GPA",
        "most_recent_position": "product research developer",
        "years_since_graduation": 0.0,
        "university": "University of Toronto",
        "companies": "",
    }),
    ("Shakoor Shaik", "mne analyst - industrials", {
        "career_objective": "",
        "skills": "java c c++ python javascript typescript sql html css r react.js node.js junit express.js p5.js flask fastapi restful api tailwind css git github firebase google cloud platform postgresql scikit-learn linux unix jira android studio intellij eclipse jupyter notebook r studio pycharm",
        "major_field_of_studies": "computer science software engineering",
        "gpa_normalized": 0.0,
        "total_work_experience_years": 1.85,
        "has_certification": 0,
        "num_certifications": 0,
        "has_career_objective": 0,
        "degree_level": "Bachelors",
        "result_type": "Other",
        "most_recent_position": "stem instructor",
        "years_since_graduation": 0.0,
        "university": "University of Toronto",
        "companies": "",
    }),
    ("Hiu Yan Kwok", "mne analyst - industrials", {
        "career_objective": "",
        "skills": "python c sql r regression modeling hypothesis testing multiple testing correction dimensionality reduction clustering reinforcement learning q-learning stochastic optimization pandas numpy matplotlib jupyter notebook git pyspark tableau",
        "major_field_of_studies": "statistics computer science",
        "gpa_normalized": 0.9775,
        "total_work_experience_years": 1.0,
        "has_certification": 1,
        "num_certifications": 1,
        "has_career_objective": 0,
        "degree_level": "Bachelors",
        "result_type": "GPA",
        "most_recent_position": "data analyst",
        "years_since_graduation": 0.0,
        "university": "University of Toronto",
        "companies": "Royal Bank of Canada Sun Life Financial",
    }),
]


def build_csv(out_path: str) -> pd.DataFrame:
    """Assemble candidate × job pairs into a DataFrame matching cleaned_resume_data.csv columns."""
    from university_ranking import compute_university_rank_features
    from company_features import compute_company_features

    ranking_dir = os.path.join(_ROOT, "data", "university_rankings")
    companies_csv = os.path.join(_ROOT, "data", "Largest_Companies.csv")

    rows = []
    for name, job_key, res in CANDIDATES:
        job = JOBS[job_key]

        # University ranking features
        uni_feats = compute_university_rank_features(res["university"], ranking_dir)

        # Company features (from candidate's past employers)
        comp_feats = compute_company_features(res.get("companies", ""), companies_csv)

        row = {
            # Text fields
            "career_objective": res["career_objective"],
            "skills": res["skills"],
            "major_field_of_studies": res["major_field_of_studies"],
            "related_skils_in_job": job["related_skils_in_job"],
            "responsibilities": job["responsibilities"],
            "job_position_name": job_key,
            "educationaL_requirements": job["educationaL_requirements"],
            "skills_required": job["skills_required"],
            # Dummy target (we are predicting this)
            "matched_score": 0.0,
            # University features
            "university_rank_score": uni_feats["university_rank_score"],
            "university_world_rank": uni_feats["university_world_rank"],
            "university_is_ranked": uni_feats["university_is_ranked"],
            # Company features
            "company_is_fortune100": comp_feats["company_is_fortune100"],
            "company_fortune100_rank_norm": comp_feats["company_fortune100_rank_norm"],
            "company_size_norm": comp_feats["company_size_norm"],
            # Numerical
            "experience_min_years": job["experience_min_years"],
            "age_min": job["age_min"],
            "age_max": job["age_max"],
            "years_since_graduation": res["years_since_graduation"],
            "gpa_normalized": res["gpa_normalized"],
            "total_work_experience_years": res["total_work_experience_years"],
            "has_certification": res["has_certification"],
            "num_certifications": res["num_certifications"],
            "has_career_objective": res["has_career_objective"],
            # Categorical
            "degree_level": res["degree_level"],
            "result_type": res["result_type"],
            "most_recent_position": res["most_recent_position"],
            # Metadata (not used by model, for display only)
            "_name": name,
            "_job": job_key,
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    print(f"Saved {len(df)} candidate-job pairs to {out_path}")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# 2.  Predict
# ══════════════════════════════════════════════════════════════════════════════

def predict(df: pd.DataFrame) -> np.ndarray:
    """Run the SBERT model on new data and return predictions."""
    cache_dir = os.path.join(_ROOT, "data", "cache")
    device = torch.device("cpu")

    # ── Load cached preprocessing artefacts ──────────────────────────────
    with open(os.path.join(cache_dir, "cat_vocab.pkl"), "rb") as f:
        cat_vocab = pickle.load(f)
    num_mean = np.load(os.path.join(cache_dir, "num_mean.npy"))
    num_std = np.load(os.path.join(cache_dir, "num_std.npy"))

    from gensim.models import Word2Vec as GensimWord2Vec
    w2v_model = GensimWord2Vec.load(os.path.join(cache_dir, "word2vec.model"))

    # ── Branch 1: numerical + categorical ────────────────────────────────
    num_arr = df[NUMERICAL_COLS].fillna(0.0).values.astype(np.float32)
    num_arr = (num_arr - num_mean) / (num_std + 1e-8)

    cat_arrs = np.stack([
        _encode_categoricals(row, cat_vocab) for _, row in df.iterrows()
    ])
    branch1 = np.concatenate([num_arr, cat_arrs], axis=1).astype(np.float32)
    branch1_dim = branch1.shape[1]
    print(f"  Branch 1 shape: {branch1.shape}")

    # ── Branch 2: SBERT embeddings ───────────────────────────────────────
    print("  Computing SBERT embeddings ...")
    sbert_parts = []
    for col in SBERT_TEXT_COLS:
        texts = df[col].fillna("").astype(str).tolist()
        emb = _compute_sbert_embeddings(texts)
        sbert_parts.append(emb)
    sbert_all = np.concatenate(sbert_parts, axis=1)

    # ── Branch 2: Word2Vec embeddings ────────────────────────────────────
    w2v_parts = []
    for col in W2V_TEXT_COLS:
        texts = df[col].fillna("").astype(str).tolist()
        emb = np.stack([_text_to_w2v_vector(t, w2v_model) for t in texts])
        w2v_parts.append(emb)
    w2v_all = np.concatenate(w2v_parts, axis=1)

    branch2 = np.concatenate([sbert_all, w2v_all], axis=1).astype(np.float32)
    branch2_dim = branch2.shape[1]
    print(f"  Branch 2 shape: {branch2.shape}")

    # ── Load model ───────────────────────────────────────────────────────
    model = ResumeMatchNet(branch1_dim=branch1_dim, branch2_dim=branch2_dim).to(device)
    ckpt = torch.load(
        os.path.join(_ROOT, "data", "primary_model", "best_model.pt"),
        map_location=device, weights_only=False,
    )
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"  Loaded SBERT model (epoch {ckpt.get('epoch', '?')})")

    # ── Forward pass ─────────────────────────────────────────────────────
    with torch.no_grad():
        b1 = torch.tensor(branch1, dtype=torch.float32)
        b2 = torch.tensor(branch2, dtype=torch.float32)
        preds = model(b1, b2).numpy()

    return preds


# ══════════════════════════════════════════════════════════════════════════════
# 3.  Main
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    csv_path = os.path.join(_ROOT, "data", "new_candidate_data.csv")

    print("Building new-data CSV ...")
    df = build_csv(csv_path)

    print("\nRunning predictions ...")
    preds = predict(df)

    # ── Print results ────────────────────────────────────────────────────
    print("\n" + "=" * 85)
    print(f"{'Candidate':<35s} {'Job Position':<35s} {'Prediction':>10s}")
    print("=" * 85)
    for i, (name, job_key, _) in enumerate(CANDIDATES):
        print(f"{name:<35s} {job_key:<35s} {preds[i]:>10.4f}")
    print("=" * 85)
    print(f"\nTotal candidates: {len(CANDIDATES)}")
