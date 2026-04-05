"""
rnn_encoder.py
--------------
Bidirectional LSTM text encoder that produces 384-dim embeddings per text field,
serving as a drop-in replacement for SBERT in the ResumeMatchNet architecture.

Architecture per text field:
    tokens → Embedding(vocab, 128) → BiLSTM(hidden=192, layers=2) → mean-pool → 384-dim

Three text fields × 384 = 1152-dim total, matching SBERT's 384×3 output.
"""

import pickle
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


# ──────────────────────────────────────────────────────────────────────────────
# Vocabulary
# ──────────────────────────────────────────────────────────────────────────────

class Vocabulary:
    """Word-level vocabulary with <pad>=0 and <unk>=1."""

    PAD_IDX = 0
    UNK_IDX = 1

    def __init__(self):
        self.word2idx = {"<pad>": 0, "<unk>": 1}
        self.idx2word = {0: "<pad>", 1: "<unk>"}

    def build(self, texts, min_count=2):
        """Build vocabulary from a list of raw text strings."""
        word_count = {}
        for text in texts:
            for word in str(text).lower().split():
                word_count[word] = word_count.get(word, 0) + 1
        for word, count in sorted(word_count.items()):
            if count >= min_count and word not in self.word2idx:
                idx = len(self.word2idx)
                self.word2idx[word] = idx
                self.idx2word[idx] = word
        return self

    def encode(self, text, max_len=None):
        """Tokenise text → list of integer indices."""
        tokens = str(text).lower().split()
        if max_len:
            tokens = tokens[:max_len]
        ids = [self.word2idx.get(w, self.UNK_IDX) for w in tokens]
        return ids if ids else [self.UNK_IDX]

    def __len__(self):
        return len(self.word2idx)

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump({"word2idx": self.word2idx, "idx2word": self.idx2word}, f)

    @classmethod
    def load(cls, path):
        vocab = cls()
        with open(path, "rb") as f:
            data = pickle.load(f)
        vocab.word2idx = data["word2idx"]
        vocab.idx2word = data["idx2word"]
        return vocab


# ──────────────────────────────────────────────────────────────────────────────
# BiLSTM Encoder
# ──────────────────────────────────────────────────────────────────────────────

class TextRNN(nn.Module):
    """
    Bidirectional LSTM text encoder.

    Input:  token IDs (B, T) + lengths (B,)
    Output: mean-pooled hidden states (B, out_dim)
            where out_dim = hidden_size × 2 = 384 by default
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 128,
        hidden_size: int = 192,
        num_layers: int = 2,
        dropout: float = 0.3,
        padding_idx: int = 0,
    ):
        super().__init__()
        self.padding_idx = padding_idx
        self.out_dim = hidden_size * 2  # bidirectional → 384

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=padding_idx)
        self.rnn = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

    def forward(self, token_ids: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        token_ids : (B, T) LongTensor
        lengths   : (B,)   LongTensor — actual sequence lengths

        Returns
        -------
        (B, 384) mean-pooled BiLSTM output
        """
        emb = self.embedding(token_ids)                          # (B, T, embed_dim)
        packed = pack_padded_sequence(
            emb, lengths.cpu().clamp(min=1), batch_first=True, enforce_sorted=False
        )
        output, _ = self.rnn(packed)
        output, _ = pad_packed_sequence(output, batch_first=True)  # (B, T', hidden*2)

        # Mean-pool over non-padding positions
        T_out = output.size(1)
        mask = (token_ids[:, :T_out] != self.padding_idx).unsqueeze(-1).float()  # (B, T', 1)
        pooled = (output * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        return pooled  # (B, 384)


# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    vocab = Vocabulary().build(["hello world", "hello test", "world test foo"])
    print(f"Vocab size: {len(vocab)}")

    rnn = TextRNN(vocab_size=len(vocab))
    print(f"Output dim: {rnn.out_dim}")
    print(f"Parameters: {sum(p.numel() for p in rnn.parameters()):,}")

    ids = torch.tensor([[1, 2, 3, 0], [2, 3, 0, 0]], dtype=torch.long)
    lens = torch.tensor([3, 2], dtype=torch.long)
    out = rnn(ids, lens)
    print(f"Output shape: {out.shape}")  # (2, 384)
