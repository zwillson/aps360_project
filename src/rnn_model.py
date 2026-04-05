"""
rnn_model.py
------------
RNN variant of ResumeMatchNet — replaces pre-computed SBERT embeddings
with a trainable Bidirectional LSTM encoder.

Architecture:
    Branch 1:  77-dim (numerical + one-hot) → FC(512) → FC(256)     [identical]
    Branch 2:  BiLSTM×3 (384 each) + W2V (300) = 1452 → FC(512) → FC(256)
    Merge:     concat(256+256) → FC(512) → Dropout → FC(256) → Dropout → FC(1) → Sigmoid

The BiLSTM is shared across all 3 text fields (career_objective,
responsibilities, educationaL_requirements) to reduce parameter count
and improve generalisation with limited training data.
"""

import torch
import torch.nn as nn

from model import _fc_block
from rnn_encoder import TextRNN


class ResumeMatchNetRNN(nn.Module):
    """
    Parameters
    ----------
    branch1_dim : dim of categorical + numerical vector (77)
    vocab_size  : RNN vocabulary size
    w2v_dim     : Word2Vec embedding size per field × 3 (300)
    embed_dim   : token embedding dimension for the BiLSTM
    rnn_hidden  : LSTM hidden size per direction (192 → 384 bidirectional)
    rnn_layers  : number of LSTM layers
    hidden_dim  : width of FC hidden layers
    dropout     : dropout rate
    """

    def __init__(
        self,
        branch1_dim: int,
        vocab_size: int,
        w2v_dim: int = 300,
        embed_dim: int = 128,
        rnn_hidden: int = 192,
        rnn_layers: int = 2,
        hidden_dim: int = 512,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.branch1_dim = branch1_dim

        # ── Branch 1: Categorical + Numerical (identical to original) ───
        self.branch1_fc = nn.Sequential(
            _fc_block(branch1_dim, hidden_dim),
            _fc_block(hidden_dim, hidden_dim // 2),
        )

        # ── Shared BiLSTM text encoder ──────────────────────────────────
        self.text_rnn = TextRNN(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            hidden_size=rnn_hidden,
            num_layers=rnn_layers,
            dropout=dropout,
        )

        # ── Branch 2: RNN outputs + Word2Vec ────────────────────────────
        branch2_dim = self.text_rnn.out_dim * 3 + w2v_dim  # 384×3 + 300 = 1452
        self.branch2_fc = nn.Sequential(
            _fc_block(branch2_dim, hidden_dim),
            _fc_block(hidden_dim, hidden_dim // 2),
        )

        # ── Merge head (identical to original) ──────────────────────────
        merged_in = (hidden_dim // 2) * 2
        self.merged_fc = nn.Sequential(
            _fc_block(merged_in, hidden_dim, dropout=dropout),
            _fc_block(hidden_dim, hidden_dim // 2, dropout=dropout),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )

        self._init_weights()

    def _init_weights(self):
        """Xavier init for linear layers; LSTM uses default PyTorch init."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, branch1, w2v, text_career, text_resp, text_edu):
        """
        Parameters
        ----------
        branch1      : (B, 77)  numerical + one-hot features
        w2v          : (B, 300) Word2Vec embeddings (3 fields × 100)
        text_career  : (token_ids (B,T1), lengths (B,))
        text_resp    : (token_ids (B,T2), lengths (B,))
        text_edu     : (token_ids (B,T3), lengths (B,))

        Returns
        -------
        (B,) predicted matched_score ∈ (0, 1)
        """
        h1 = self.branch1_fc(branch1)

        # Encode each text field with the shared BiLSTM
        career_emb = self.text_rnn(text_career[0], text_career[1])  # (B, 384)
        resp_emb   = self.text_rnn(text_resp[0],   text_resp[1])    # (B, 384)
        edu_emb    = self.text_rnn(text_edu[0],     text_edu[1])     # (B, 384)

        branch2_in = torch.cat([career_emb, resp_emb, edu_emb, w2v], dim=1)  # (B, 1452)
        h2 = self.branch2_fc(branch2_in)

        h = torch.cat([h1, h2], dim=1)
        return self.merged_fc(h).squeeze(1)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def __repr__(self):
        base = super().__repr__()
        return f"{base}\n\nTotal trainable parameters: {self.count_parameters():,}"


# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    B1_DIM = 15 + 6 + 5 + 51   # 77
    VOCAB  = 5000

    model = ResumeMatchNetRNN(branch1_dim=B1_DIM, vocab_size=VOCAB)
    print(model)

    # Dummy forward pass
    B = 4
    b1  = torch.randn(B, B1_DIM)
    w2v = torch.randn(B, 300)
    tc  = (torch.randint(0, VOCAB, (B, 20)), torch.tensor([20, 15, 10, 5]))
    tr  = (torch.randint(0, VOCAB, (B, 30)), torch.tensor([30, 25, 20, 10]))
    te  = (torch.randint(0, VOCAB, (B, 15)), torch.tensor([15, 12, 8, 5]))
    out = model(b1, w2v, tc, tr, te)
    print(f"\nOutput shape: {out.shape}  values: {out.detach().numpy().round(4)}")
