"""
model.py
--------
Multi-branch neural network for resume-job matching (regression).

Architecture (matches the design diagram):

    ┌──────────────────────────────────┐
    │  Branch 1: Categorical + Numeric │
    │  ─────────────────────────────── │
    │  [One-Hot encoded categories]    │
    │  [Normalised numeric features]   │
    │         ↓  concatenate           │
    │   Flatten → FC(512) → BN → ReLU  │
    │           → FC(256) → BN → ReLU  │
    └──────────────────┬───────────────┘
                       │  Combined Vector
    ┌──────────────────▼───────────────┐
    │                                  │
    │  Branch 2: Text Embeddings       │
    │  ─────────────────────────────── │
    │  [SBERT: career_obj, respon,     │
    │           edu_req]   (384 × 3)   │
    │  [Word2Vec: skills, skills_req,  │
    │           related_skills](100×3) │
    │         ↓  concatenate (1452d)   │
    │   Flatten → FC(512) → BN → ReLU  │
    │           → FC(256) → BN → ReLU  │
    └──────────────────┬───────────────┘
                       │  Fully Connected repr
    ┌──────────────────▼───────────────┐
    │  Merged Layer + Hidden Layers    │
    │  ─────────────────────────────── │
    │   concat → FC(512) → BN → ReLU   │
    │           → Dropout(0.3)         │
    │           → FC(256) → BN → ReLU  │
    │           → Dropout(0.3)         │
    │           → FC(1) → Sigmoid      │
    └──────────────────────────────────┘
                   ↓
            matched_score ∈ (0, 1)

Loss: MSELoss  (regression task)
"""

import torch
import torch.nn as nn


# ──────────────────────────────────────────────────────────────────────────────
# Building blocks
# ──────────────────────────────────────────────────────────────────────────────

def _fc_block(in_dim: int, out_dim: int, dropout: float = 0.0) -> nn.Sequential:
    """Fully-connected block: Linear → BatchNorm → ReLU [→ Dropout]."""
    layers = [
        nn.Linear(in_dim, out_dim),
        nn.BatchNorm1d(out_dim),
        nn.ReLU(inplace=True),
    ]
    if dropout > 0:
        layers.append(nn.Dropout(dropout))
    return nn.Sequential(*layers)


# ──────────────────────────────────────────────────────────────────────────────
# Primary model
# ──────────────────────────────────────────────────────────────────────────────

class ResumeMatchNet(nn.Module):
    """
    Parameters
    ----------
    branch1_dim  : dimension of the categorical+numerical feature vector
    branch2_dim  : dimension of the concatenated text embedding vector
    hidden_dim   : width of hidden layers (default 512 / 256)
    dropout      : dropout rate in merged layers
    """

    def __init__(
        self,
        branch1_dim: int,
        branch2_dim: int,
        hidden_dim: int = 512,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.branch1_dim = branch1_dim
        self.branch2_dim = branch2_dim

        # ── Branch 1: Categorical + Numerical ───────────────────────────
        # Flatten is implicit (input is already a vector)
        self.branch1_fc = nn.Sequential(
            _fc_block(branch1_dim, hidden_dim),
            _fc_block(hidden_dim, hidden_dim // 2),
        )

        # ── Branch 2: Text embeddings ────────────────────────────────────
        self.branch2_fc = nn.Sequential(
            _fc_block(branch2_dim, hidden_dim),
            _fc_block(hidden_dim, hidden_dim // 2),
        )

        # ── Merged layers ─────────────────────────────────────────────────
        merged_in = (hidden_dim // 2) * 2  # concat both branch outputs
        self.merged_fc = nn.Sequential(
            _fc_block(merged_in, hidden_dim, dropout=dropout),
            _fc_block(hidden_dim, hidden_dim // 2, dropout=dropout),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),   # output in (0, 1) — matches matched_score range
        )

        self._init_weights()

    def _init_weights(self):
        """Xavier initialisation for all linear layers."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, branch1: torch.Tensor, branch2: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        branch1 : (B, branch1_dim)  — categorical + numerical features
        branch2 : (B, branch2_dim)  — SBERT + Word2Vec embeddings

        Returns
        -------
        (B,) tensor of predicted matched_scores in (0, 1)
        """
        h1 = self.branch1_fc(branch1)   # (B, hidden_dim // 2)
        h2 = self.branch2_fc(branch2)   # (B, hidden_dim // 2)
        h  = torch.cat([h1, h2], dim=1) # (B, hidden_dim)
        out = self.merged_fc(h)          # (B, 1)
        return out.squeeze(1)            # (B,)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def __repr__(self):
        base = super().__repr__()
        return f"{base}\n\nTotal trainable parameters: {self.count_parameters():,}"


# ──────────────────────────────────────────────────────────────────────────────
# Quick sanity check
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Dimensions matching the dataset:
    #   branch1: 15 numerical + 6(degree) + 5(result) + 51(job) = 77
    #     numerical: 9 base + 3 university ranking + 3 Fortune100 company = 15
    #   branch2: 384*3 (SBERT) + 100*3 (Word2Vec) = 1452
    B1_DIM = 15 + 6 + 5 + 51  # 77
    B2_DIM = 384 * 3 + 100 * 3  # 1452

    model = ResumeMatchNet(branch1_dim=B1_DIM, branch2_dim=B2_DIM)
    print(model)

    # Forward pass with random data
    b1 = torch.randn(8, B1_DIM)
    b2 = torch.randn(8, B2_DIM)
    out = model(b1, b2)
    print(f"\nOutput shape: {out.shape}  values: {out.detach().numpy().round(4)}")
