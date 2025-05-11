# jigsaw.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class JigsawSolver(nn.Module):
    """
    Given permuted sketch patches and intact photo patches,
    predicts each sketch‐patch's original index.
    """
    def __init__(self, dv: int, num_patches: int):
        super().__init__()
        # simple 2‐layer MLP: input size = 2*dv, hidden = 2*dv, output = num_patches
        self.head = nn.Sequential(
            nn.Linear(2*dv, 2*dv),
            nn.ReLU(),
            nn.Linear(2*dv, num_patches)
        )

    def forward(self, sk_patches: torch.Tensor, ph_patches: torch.Tensor):
        """
        sk_patches: (B, P, dv)  — permuted sketch patch features
        ph_patches: (B, P, dv)  — intact photo patch features
        returns   : (B, P, P)    — logits over original indices
        """
        # concatenate along feature dim
        x = torch.cat([sk_patches, ph_patches], dim=-1)  # (B, P, 2*dv)
        return self.head(x)                              # (B, P, P)


def compute_jigsaw_loss(
    sk_patches: torch.Tensor,
    ph_patches: torch.Tensor,
    solver: JigsawSolver
) -> torch.Tensor:
    """
    Cross‐entropy loss for re‐assembling permuted sketch patches.
    sk_patches: (B, P, dv)  — prompt‐injected sketch patches
    ph_patches: (B, P, dv)  — prompt‐injected photo patches
    solver    : JigsawSolver
    """
    B, P, dv = sk_patches.shape
    # random permutation of the P positions
    perm = torch.randperm(P, device=sk_patches.device)
    sk_perm = sk_patches[:, perm, :]             # (B, P, dv)

    logits = solver(sk_perm, ph_patches)          # (B, P, P)
    logits = logits.reshape(B * P, P)             # (B*P, P)

    # target[i] = the original index of the i-th permuted patch
    target = perm.unsqueeze(0).expand(B, P).reshape(B * P)  # (B*P,)

    return F.cross_entropy(logits, target)
