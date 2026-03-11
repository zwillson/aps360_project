"""
dataset.py
----------
PyTorch Dataset for the multi-branch resume-matching model.

Two feature branches (mirrors the architecture diagram):

Branch 1 — Categorical + Numerical
    Numerical scalars:  experience_min_years, age_min, age_max,
                        years_since_graduation, gpa_normalized,
                        total_work_experience_years, has_certification,
                        num_certifications, has_career_objective,
                        university_rank_score, university_world_rank,
                        university_is_ranked,
                        company_is_fortune100, company_fortune100_rank_norm,
                        company_size_norm
    One-hot categorical: degree_level (6 classes), result_type (5 classes),
                         job_position_name (top-50 + other = 51 classes)
    → concatenated into a single float tensor (branch1_dim = 15+6+5+51 = 77)

Branch 2 — Text embeddings
    SBERT (384-dim each):
        career_objective, responsibilities, educationaL_requirements
    Word2Vec (100-dim each):
        skills, skills_required, related_skils_in_job
    → concatenated into a float tensor (branch2_dim = 384*3 + 100*3 = 1452)

Usage
-----
    # First call (slow — trains Word2Vec and caches SBERT embeddings):
    ds = ResumeDataset('../data/cleaned_resume_data.csv', cache_dir='../data/cache')

    # Subsequent calls (fast — loads from cache):
    ds = ResumeDataset('../data/cleaned_resume_data.csv', cache_dir='../data/cache')

    # Use with DataLoader:
    loader = DataLoader(ds, batch_size=64, shuffle=True)
    for branch1, branch2, target in loader:
        ...
"""

import os
import pickle
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


# ──────────────────────────────────────────────────────────────────────────────
# Column definitions
# ──────────────────────────────────────────────────────────────────────────────

NUMERICAL_COLS = [
    "experience_min_years",
    "age_min",
    "age_max",
    "years_since_graduation",
    "gpa_normalized",
    "total_work_experience_years",
    "has_certification",
    "num_certifications",
    "has_career_objective",
    # University ranking features (CWUR/Shanghai/Times)
    "university_rank_score",   # normalized 0-1: quality score averaged across CWUR/Shanghai/Times
    "university_world_rank",   # normalized 0-1: best world rank (1=rank1, 0=rank1000+)
    "university_is_ranked",    # binary: 1 if any listed institution appears in a global ranking
    # Fortune 100 company features (Largest_Companies.csv)
    "company_is_fortune100",        # binary: 1 if any employer matched a Fortune 100 company
    "company_fortune100_rank_norm", # normalized 0-1: best employer rank (1=rank#1/Walmart)
    "company_size_norm",            # normalized 0-1: log employee count of best matched employer
]

CATEGORICAL_COLS = ["degree_level", "result_type", "job_position_name"]

SBERT_TEXT_COLS = [
    "career_objective",
    "responsibilities",
    "educationaL_requirements",
]

W2V_TEXT_COLS = [
    "skills",
    "skills_required",
    "related_skils_in_job",
]

TARGET = "matched_score"

SBERT_DIM = 384
W2V_DIM   = 100
TOP_K_JOBS = 50


# ──────────────────────────────────────────────────────────────────────────────
# Categorical encoding helpers
# ──────────────────────────────────────────────────────────────────────────────

DEGREE_LEVELS   = ["Bachelors", "Masters", "PhD", "Diploma/Certificate", "Other", "Unknown"]
RESULT_TYPES    = ["GPA", "CGPA", "Percentage", "Grade", "Other"]


def _build_cat_vocab(df: pd.DataFrame, top_k_jobs: int = TOP_K_JOBS) -> dict:
    """Build categorical vocabularies from the dataframe."""
    top_jobs = df["job_position_name"].value_counts().nlargest(top_k_jobs).index.tolist()
    return {
        "degree_level":    {v: i for i, v in enumerate(DEGREE_LEVELS)},
        "result_type":     {v: i for i, v in enumerate(RESULT_TYPES)},
        "job_position_name": {v: i for i, v in enumerate(top_jobs + ["other"])},
    }


def _one_hot(idx: int, size: int) -> np.ndarray:
    v = np.zeros(size, dtype=np.float32)
    v[idx] = 1.0
    return v


def _encode_categoricals(row: pd.Series, vocab: dict) -> np.ndarray:
    parts = []
    for col in CATEGORICAL_COLS:
        v_map = vocab[col]
        val = str(row[col]).strip()
        idx = v_map.get(val, v_map.get("other", v_map.get("Other", len(v_map) - 1)))
        parts.append(_one_hot(idx, len(v_map)))
    return np.concatenate(parts)


# ──────────────────────────────────────────────────────────────────────────────
# SBERT embeddings
# ──────────────────────────────────────────────────────────────────────────────

def _compute_sbert_embeddings(texts: list, model_name: str = "all-MiniLM-L6-v2") -> np.ndarray:
    """
    Encode a list of strings with SBERT. Returns array of shape (N, SBERT_DIM).
    Empty strings get a zero vector.
    """
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_name)
    # Replace empty strings with a sentinel so SBERT handles them gracefully
    safe_texts = [t if t.strip() else "empty" for t in texts]
    embeddings = model.encode(safe_texts, batch_size=128, show_progress_bar=True,
                               convert_to_numpy=True)
    # Zero-out rows that were originally empty
    for i, t in enumerate(texts):
        if not t.strip():
            embeddings[i] = 0.0
    return embeddings.astype(np.float32)


# ──────────────────────────────────────────────────────────────────────────────
# Word2Vec embeddings
# ──────────────────────────────────────────────────────────────────────────────

def _train_word2vec(sentences: list, dim: int = W2V_DIM) -> object:
    """Train a Word2Vec model on a list of tokenised sentences (list of list of str)."""
    from gensim.models import Word2Vec
    print(f"  Training Word2Vec on {len(sentences)} sentences ...")
    model = Word2Vec(
        sentences=sentences,
        vector_size=dim,
        window=5,
        min_count=2,
        workers=4,
        epochs=10,
        seed=42,
    )
    return model


def _text_to_w2v_vector(text: str, w2v_model, dim: int = W2V_DIM) -> np.ndarray:
    """Average Word2Vec vectors for all tokens in text. Returns zero vec if OOV."""
    if not text.strip():
        return np.zeros(dim, dtype=np.float32)
    tokens = text.lower().split()
    vecs = []
    for tok in tokens:
        if tok in w2v_model.wv:
            vecs.append(w2v_model.wv[tok])
    if not vecs:
        return np.zeros(dim, dtype=np.float32)
    return np.mean(vecs, axis=0).astype(np.float32)


# ──────────────────────────────────────────────────────────────────────────────
# Main Dataset class
# ──────────────────────────────────────────────────────────────────────────────

class ResumeDataset(Dataset):
    """
    Parameters
    ----------
    csv_path    : path to cleaned_resume_data.csv
    cache_dir   : directory to store/load precomputed embeddings
    fit         : if True, recompute and re-save Word2Vec + SBERT embeddings
    """

    def __init__(self, csv_path: str, cache_dir: str = None, fit: bool = False):
        self.csv_path  = csv_path
        self.cache_dir = cache_dir or os.path.join(os.path.dirname(csv_path), "cache")
        os.makedirs(self.cache_dir, exist_ok=True)

        print(f"Loading {csv_path} ...")
        df = pd.read_csv(csv_path)
        self.n = len(df)

        # ── Build categorical vocabulary ─────────────────────────────────
        vocab_path = os.path.join(self.cache_dir, "cat_vocab.pkl")
        if fit or not os.path.exists(vocab_path):
            self.vocab = _build_cat_vocab(df, top_k_jobs=TOP_K_JOBS)
            with open(vocab_path, "wb") as f:
                pickle.dump(self.vocab, f)
            print(f"  Saved categorical vocab to {vocab_path}")
        else:
            with open(vocab_path, "rb") as f:
                self.vocab = pickle.load(f)
            print(f"  Loaded categorical vocab from {vocab_path}")

        # ── Branch 1: Numerical + Categorical ────────────────────────────
        b1_path = os.path.join(self.cache_dir, "branch1.npy")
        if fit or not os.path.exists(b1_path):
            print("  Building Branch 1 (numerical + categorical) ...")
            num_arr  = df[NUMERICAL_COLS].fillna(0.0).values.astype(np.float32)
            cat_arrs = np.stack([
                _encode_categoricals(row, self.vocab) for _, row in df.iterrows()
            ])
            # Normalise numerical features (z-score, computed on this split)
            self.num_mean = num_arr.mean(axis=0)
            self.num_std  = num_arr.std(axis=0) + 1e-8
            np.save(os.path.join(self.cache_dir, "num_mean.npy"), self.num_mean)
            np.save(os.path.join(self.cache_dir, "num_std.npy"),  self.num_std)
            num_arr = (num_arr - self.num_mean) / self.num_std
            branch1 = np.concatenate([num_arr, cat_arrs], axis=1)
            np.save(b1_path, branch1)
            print(f"  Branch 1 shape: {branch1.shape}  saved to {b1_path}")
        else:
            branch1 = np.load(b1_path)
            self.num_mean = np.load(os.path.join(self.cache_dir, "num_mean.npy"))
            self.num_std  = np.load(os.path.join(self.cache_dir, "num_std.npy"))
            print(f"  Loaded Branch 1 from {b1_path}  shape={branch1.shape}")

        self.branch1 = branch1
        self.branch1_dim = branch1.shape[1]

        # ── Branch 2: SBERT embeddings ────────────────────────────────────
        sbert_path = os.path.join(self.cache_dir, "sbert_embeddings.npy")
        if fit or not os.path.exists(sbert_path):
            print("  Computing SBERT embeddings (may take a few minutes) ...")
            sbert_parts = []
            for col in SBERT_TEXT_COLS:
                texts = df[col].fillna("").astype(str).tolist()
                print(f"    SBERT: {col}")
                emb = _compute_sbert_embeddings(texts)
                sbert_parts.append(emb)
            sbert_all = np.concatenate(sbert_parts, axis=1)  # (N, 384*3)
            np.save(sbert_path, sbert_all)
            print(f"  SBERT shape: {sbert_all.shape}  saved to {sbert_path}")
        else:
            sbert_all = np.load(sbert_path)
            print(f"  Loaded SBERT embeddings from {sbert_path}  shape={sbert_all.shape}")

        # ── Branch 2: Word2Vec embeddings ─────────────────────────────────
        w2v_model_path = os.path.join(self.cache_dir, "word2vec.model")
        w2v_emb_path   = os.path.join(self.cache_dir, "w2v_embeddings.npy")
        if fit or not os.path.exists(w2v_model_path):
            print("  Training Word2Vec ...")
            # Combine all text columns to build vocabulary
            all_sentences = []
            for col in W2V_TEXT_COLS:
                for text in df[col].fillna("").astype(str):
                    tokens = text.lower().split()
                    if tokens:
                        all_sentences.append(tokens)
            w2v_model = _train_word2vec(all_sentences, dim=W2V_DIM)
            w2v_model.save(w2v_model_path)
            print(f"  Word2Vec saved to {w2v_model_path}")
        else:
            from gensim.models import Word2Vec
            w2v_model = Word2Vec.load(w2v_model_path)
            print(f"  Loaded Word2Vec from {w2v_model_path}")

        if fit or not os.path.exists(w2v_emb_path):
            print("  Computing Word2Vec embeddings ...")
            w2v_parts = []
            for col in W2V_TEXT_COLS:
                texts = df[col].fillna("").astype(str).tolist()
                emb = np.stack([_text_to_w2v_vector(t, w2v_model) for t in texts])
                w2v_parts.append(emb)
            w2v_all = np.concatenate(w2v_parts, axis=1)  # (N, 100*3)
            np.save(w2v_emb_path, w2v_all)
            print(f"  W2V shape: {w2v_all.shape}  saved to {w2v_emb_path}")
        else:
            w2v_all = np.load(w2v_emb_path)
            print(f"  Loaded W2V embeddings from {w2v_emb_path}  shape={w2v_all.shape}")

        # ── Concatenate both embedding sources into Branch 2 ──────────────
        self.branch2 = np.concatenate([sbert_all, w2v_all], axis=1).astype(np.float32)
        self.branch2_dim = self.branch2.shape[1]
        print(f"  Branch 2 shape: {self.branch2.shape}")

        # ── Target ────────────────────────────────────────────────────────
        self.targets = df[TARGET].values.astype(np.float32)

        print(f"\nDataset ready: {self.n} samples  "
              f"branch1_dim={self.branch1_dim}  branch2_dim={self.branch2_dim}")

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        return (
            torch.tensor(self.branch1[idx], dtype=torch.float32),
            torch.tensor(self.branch2[idx], dtype=torch.float32),
            torch.tensor(self.targets[idx], dtype=torch.float32),
        )


# ──────────────────────────────────────────────────────────────────────────────
# Convenience: split dataset indices
# ──────────────────────────────────────────────────────────────────────────────

def split_dataset(dataset: ResumeDataset, train=0.70, val=0.10, test=0.20, seed=42):
    """Return train/val/test Subset objects."""
    from torch.utils.data import Subset
    n = len(dataset)
    idx = np.random.default_rng(seed).permutation(n)
    n_train = int(n * train)
    n_val   = int(n * val)
    train_idx = idx[:n_train]
    val_idx   = idx[n_train: n_train + n_val]
    test_idx  = idx[n_train + n_val:]
    return (
        Subset(dataset, train_idx),
        Subset(dataset, val_idx),
        Subset(dataset, test_idx),
    )


# ──────────────────────────────────────────────────────────────────────────────
# Quick test
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import os
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ds = ResumeDataset(
        csv_path=os.path.join(base, "data", "cleaned_resume_data.csv"),
        cache_dir=os.path.join(base, "data", "cache"),
        fit=True,
    )
    b1, b2, y = ds[0]
    print(f"b1 shape: {b1.shape}")
    print(f"b2 shape: {b2.shape}")
    print(f"target:   {y.item():.4f}")
