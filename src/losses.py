"""
Imbalance-aware and boundary-aware losses for tampered-text localization.

Implements the supervision from the proposal (Section 3,
"Imbalance-Aware and Boundary-Aware Supervision"):

    L_T = L_CE(P, Y) + lambda_dice * L_Dice(P, Y) + lambda_bound * L_bound(B_P, B_Y)

Conventions follow the DocTamper / DTD pipeline, where the model produces
2-channel logits (background=0, tampered=1) and the ground-truth mask uses an
ignore label (100) for uncertain pixels. The author of DocTamper recommends:

    gt[gt < 8]   = 0     # background
    gt[gt > 160] = 1     # tampered
    gt[gt > 1]   = 100   # uncertain -> ignored
    loss = nn.CrossEntropyLoss(ignore_index=100)(pred, gt)

The proposal writes "BCE"; because the DTD head is 2-channel, we use the
multi-class equivalent (CrossEntropyLoss with an ignore index). Dice and the
boundary term are computed on the tampered-class probability and respect the
same ignore region.

NOTE: not executed on the dev machine (no GPU). Written to standard PyTorch
conventions; verify tensor shapes on the first Colab run.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

IGNORE_INDEX = 100


def _foreground_prob(logits: torch.Tensor) -> torch.Tensor:
    """Probability of the 'tampered' class (channel 1) from 2-channel logits.

    logits: (N, 2, H, W) -> returns (N, H, W) in [0, 1].
    """
    return torch.softmax(logits, dim=1)[:, 1]


def soft_dice_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    ignore_index: int = IGNORE_INDEX,
    eps: float = 1.0,
) -> torch.Tensor:
    """Soft Dice loss on the tampered class, ignoring 'uncertain' pixels.

    logits: (N, 2, H, W); target: (N, H, W) with values in {0, 1, ignore_index}.
    """
    prob = _foreground_prob(logits)               # (N, H, W)
    valid = (target != ignore_index).float()      # (N, H, W)
    fg = (target == 1).float() * valid
    prob = prob * valid
    dims = (1, 2)
    inter = (prob * fg).sum(dims)
    denom = prob.sum(dims) + fg.sum(dims)
    dice = (2.0 * inter + eps) / (denom + eps)
    return (1.0 - dice).mean()


def _morphological_gradient(x: torch.Tensor, kernel: int = 3) -> torch.Tensor:
    """Boundary map of a (N, 1, H, W) prob/binary map via dilation - erosion."""
    pad = kernel // 2
    dil = F.max_pool2d(x, kernel, stride=1, padding=pad)
    ero = -F.max_pool2d(-x, kernel, stride=1, padding=pad)
    return (dil - ero).clamp(0.0, 1.0)


def boundary_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    ignore_index: int = IGNORE_INDEX,
    kernel: int = 3,
    eps: float = 1.0,
) -> torch.Tensor:
    """Dice between predicted and GT boundary maps (character / glyph edges).

    B_P and B_Y are the morphological gradients of the predicted tampered
    probability and the GT tampered mask, matching the proposal's notation
    L_bound(B_P, B_Y). Penalising boundary mismatch sharpens stroke edges.
    """
    prob = _foreground_prob(logits).unsqueeze(1)              # (N, 1, H, W)
    valid = (target != ignore_index).float().unsqueeze(1)
    fg = ((target == 1).float() * valid.squeeze(1)).unsqueeze(1)
    b_pred = _morphological_gradient(prob * valid, kernel)
    b_gt = _morphological_gradient(fg, kernel)
    dims = (1, 2, 3)
    inter = (b_pred * b_gt).sum(dims)
    denom = b_pred.sum(dims) + b_gt.sum(dims)
    dice = (2.0 * inter + eps) / (denom + eps)
    return (1.0 - dice).mean()


class CombinedTamperLoss(nn.Module):
    """L_CE(ignore=100) + lambda_dice * Dice + lambda_bound * Boundary.

    Returns (total_loss, components_dict) so the individual terms can be logged
    for the ablation study described in the proposal (Section 5).
    """

    def __init__(
        self,
        lambda_dice: float = 1.0,
        lambda_bound: float = 0.5,
        ignore_index: int = IGNORE_INDEX,
        ce_weight: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.ce = nn.CrossEntropyLoss(ignore_index=ignore_index, weight=ce_weight)
        self.lambda_dice = lambda_dice
        self.lambda_bound = lambda_bound
        self.ignore_index = ignore_index

    def forward(self, logits: torch.Tensor, target: torch.Tensor):
        target = target.long()
        ce = self.ce(logits, target)
        dice = soft_dice_loss(logits, target, self.ignore_index)
        bound = boundary_loss(logits, target, self.ignore_index)
        total = ce + self.lambda_dice * dice + self.lambda_bound * bound
        components = {
            "ce": ce.detach(),
            "dice": dice.detach(),
            "boundary": bound.detach(),
            "total": total.detach(),
        }
        return total, components
