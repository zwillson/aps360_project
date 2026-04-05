"""
rnn_dataset.py
--------------
PyTorch Dataset for the RNN variant of ResumeMatchNet.

Same branch structure as dataset.py, but replaces pre-computed SBERT
embeddings with tokenised text sequences processed by a BiLSTM at train time.

Branch 1:  77-dim  (15 numerical + one-hot categorical) — identical
Branch 2:  3 × BiLSTM → 384-dim each  +  3 × Word2Vec 100-dim  =  1452-dim total
"""

import os
import pickle
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

from dataset import (
    NUMERICAL_COLS, CATEGORICAL_COLS, SBERT_TEXT_COLS, W2V_TEXT_COLS,
    TARGET, W2V_DIM, TOP_K_JOBS,
    _build_cat_vocab, _encode_categoricals,
    _train_word2vec, _text_to_w2v_vector,
)
from rnn_encoder import Vocabulary

MAX_SEQ_LEN = 256


class RNNResumeDataset(Dataset):
    """
    Parameters
    ----------
    csv_path    : path to cleaned_resume_data.csv
    cache_dir   : directory for cached embeddings / vocab
    fit         : if True, rebuild vocab and W2V from scratch
    max_seq_len : truncate text sequences to this many tokens
    """

    def __init__(self, csv_path: str, cache_dir: str = None,
                 fit: bool = False, max_seq_len: int = MAX_SEQ_LEN):
        self.cache_dir = cache_dir or os.path.join(os.path.dirname(csv_path), "cache")
        os.makedirs(self.cache_dir, exist_ok=True)

        print(f"Loading {csv_path} ...")
        df = pd.read_csv(csv_path)
        self.n = len(df)

        # ── Categorical vocabulary ───────────────────────────────────────
        vocab_path = os.path.join(self.cache_dir, "cat_vocab.pkl")
        if fit or not os.path.exists(vocab_path):
            self.cat_vocab = _build_cat_vocab(df, top_k_jobs=TOP_K_JOBS)
            with open(vocab_path, "wb") as f:
                pickle.dump(self.cat_vocab, f)
        else:
            with open(vocab_path, "rb") as f:
                self.cat_vocab = pickle.load(f)

        # ── Branch 1: Numerical + Categorical (identical to original) ───
        b1_path = os.path.join(self.cache_dir, "branch1.npy")
        if fit or not os.path.exists(b1_path):
            print("  Building Branch 1 (numerical + categorical) ...")
            num_arr = df[NUMERICAL_COLS].fillna(0.0).values.astype(np.float32)
            cat_arrs = np.stack([
                _encode_categoricals(row, self.cat_vocab) for _, row in df.iterrows()
            ])
            self.num_mean = num_arr.mean(axis=0)
            self.num_std  = num_arr.std(axis=0) + 1e-8
            np.save(os.path.join(self.cache_dir, "num_mean.npy"), self.num_mean)
            np.save(os.path.join(self.cache_dir, "num_std.npy"),  self.num_std)
            num_arr = (num_arr - self.num_mean) / self.num_std
            branch1 = np.concatenate([num_arr, cat_arrs], axis=1)
            np.save(b1_path, branch1)
            print(f"  Branch 1 shape: {branch1.shape}")
        else:
            branch1 = np.load(b1_path)
            self.num_mean = np.load(os.path.join(self.cache_dir, "num_mean.npy"))
            self.num_std  = np.load(os.path.join(self.cache_dir, "num_std.npy"))
            print(f"  Loaded Branch 1: shape={branch1.shape}")
        self.branch1 = branch1
        self.branch1_dim = branch1.shape[1]

        # ── RNN vocabulary ───────────────────────────────────────────────
        rnn_vocab_path = os.path.join(self.cache_dir, "rnn_vocab.pkl")
        if fit or not os.path.exists(rnn_vocab_path):
            print("  Building RNN vocabulary ...")
            all_texts = []
            for col in SBERT_TEXT_COLS:
                all_texts.extend(df[col].fillna("").astype(str).tolist())
            self.vocab = Vocabulary().build(all_texts, min_count=2)
            self.vocab.save(rnn_vocab_path)
            print(f"  Vocabulary size: {len(self.vocab)}")
        else:
            self.vocab = Vocabulary.load(rnn_vocab_path)
            print(f"  Loaded RNN vocabulary: {len(self.vocab)} tokens")

        # ── Tokenise the 3 SBERT text fields ─────────────────────────────
        print("  Tokenising text fields ...")
        self.text_tokens = []   # [col_idx][sample_idx] → LongTensor
        self.text_lengths = []  # [col_idx][sample_idx] → int
        for col in SBERT_TEXT_COLS:
            texts = df[col].fillna("").astype(str).tolist()
            col_tokens, col_lengths = [], []
            for t in texts:
                ids = self.vocab.encode(t, max_len=max_seq_len)
                col_tokens.append(torch.tensor(ids, dtype=torch.long))
                col_lengths.append(len(ids))
            self.text_tokens.append(col_tokens)
            self.text_lengths.append(col_lengths)

        # ── Word2Vec embeddings (identical to original) ──────────────────
        w2v_model_path = os.path.join(self.cache_dir, "word2vec.model")
        w2v_emb_path   = os.path.join(self.cache_dir, "w2v_embeddings.npy")
        if fit or not os.path.exists(w2v_model_path):
            print("  Training Word2Vec ...")
            all_sentences = []
            for col in W2V_TEXT_COLS:
                for text in df[col].fillna("").astype(str):
                    tokens = text.lower().split()
                    if tokens:
                        all_sentences.append(tokens)
            w2v_model = _train_word2vec(all_sentences, dim=W2V_DIM)
            w2v_model.save(w2v_model_path)
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
            w2v_all = np.concatenate(w2v_parts, axis=1)
            np.save(w2v_emb_path, w2v_all)
        else:
            w2v_all = np.load(w2v_emb_path)
            print(f"  Loaded W2V embeddings: shape={w2v_all.shape}")
        self.w2v = w2v_all.astype(np.float32)

        # ── Targets ──────────────────────────────────────────────────────
        self.targets = df[TARGET].values.astype(np.float32)

        print(f"\nRNN Dataset ready: {self.n} samples  "
              f"branch1_dim={self.branch1_dim}  vocab={len(self.vocab)}  "
              f"w2v_dim={self.w2v.shape[1]}")

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        return (
            torch.tensor(self.branch1[idx], dtype=torch.float32),
            torch.tensor(self.w2v[idx], dtype=torch.float32),
            self.text_tokens[0][idx],   # career_objective  tokens
            self.text_tokens[1][idx],   # responsibilities  tokens
            self.text_tokens[2][idx],   # edu_requirements  tokens
            torch.tensor(self.targets[idx], dtype=torch.float32),
        )


# ──────────────────────────────────────────────────────────────────────────────
# Collate function (pads variable-length sequences per batch)
# ──────────────────────────────────────────────────────────────────────────────

def rnn_collate_fn(batch):
    """
    Returns
    -------
    branch1      : (B, 77)
    w2v          : (B, 300)
    text_career  : (padded_ids (B,T1), lengths (B,))
    text_resp    : (padded_ids (B,T2), lengths (B,))
    text_edu     : (padded_ids (B,T3), lengths (B,))
    targets      : (B,)
    """
    branch1 = torch.stack([b[0] for b in batch])
    w2v     = torch.stack([b[1] for b in batch])
    targets = torch.stack([b[5] for b in batch])

    text_fields = []
    for i in range(3):
        tokens_list = [b[2 + i] for b in batch]
        lengths = torch.tensor([len(t) for t in tokens_list], dtype=torch.long)
        padded  = pad_sequence(tokens_list, batch_first=True, padding_value=0)
        text_fields.append((padded, lengths))

    return branch1, w2v, text_fields[0], text_fields[1], text_fields[2], targets


# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ds = RNNResumeDataset(
        csv_path=os.path.join(base, "data", "cleaned_resume_data.csv"),
        cache_dir=os.path.join(base, "data", "cache"),
        fit=True,
    )
    b1, w2v, tc, tr, te, y = ds[0]
    print(f"branch1: {b1.shape}  w2v: {w2v.shape}  "
          f"career_tokens: {tc.shape}  target: {y.item():.4f}")
