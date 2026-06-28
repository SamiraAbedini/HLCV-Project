"""
Text-prior fusion module for the DTD decoder.

Implements F_hat_l = phi_l([F_l, M_l]) from the proposal: resize the binary
text prior to a decoder feature's resolution, concatenate channel-wise, and
apply a lightweight conv block. The fusion is LEARNABLE (soft guidance), so the
model is not hard-limited to OCR regions when text detection is imperfect.

The module is intentionally generic: it takes any decoder feature map and the
text mask, so it can be inserted at one or more decoder stages of DTD without
changing the RGB-frequency backbone. See docs/INTEGRATION.md for where to hook
it inside the DocTamper model.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class TextPriorFusion(nn.Module):
    """Fuse a binary text prior into a decoder feature map.

    Parameters
    ----------
    in_channels : int
        Channel count of the decoder feature ``F_l``.
    mid_channels : int, optional
        Hidden width of the fusion conv block (defaults to ``in_channels``).
    residual : bool
        If True, add the fused result back to the input feature so the prior is
        a refinement rather than a replacement (keeps the original signal).
    """

    def __init__(
        self,
        in_channels: int,
        mid_channels: int | None = None,
        residual: bool = True,
    ) -> None:
        super().__init__()
        mid = mid_channels or in_channels
        self.block = nn.Sequential(
            nn.Conv2d(in_channels + 1, mid, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, in_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels),
        )
        self.act = nn.ReLU(inplace=True)
        self.residual = residual

    def forward(self, feat: torch.Tensor, text_mask: torch.Tensor) -> torch.Tensor:
        """feat: (N, C, h, w); text_mask: (N, 1, H, W) or (N, H, W) in {0, 1}."""
        if text_mask.dim() == 3:
            text_mask = text_mask.unsqueeze(1)
        # nearest keeps the mask binary; align to the feature's spatial size.
        m = F.interpolate(text_mask.float(), size=feat.shape[-2:], mode="nearest")
        x = torch.cat([feat, m], dim=1)
        out = self.block(x)
        if self.residual:
            out = out + feat
        return self.act(out)
