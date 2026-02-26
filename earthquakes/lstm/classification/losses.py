"""Classification loss functions for ordinal/magnitude prediction."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.nn import functional


class FocalLoss(nn.Module):
    """Focal loss for imbalanced classification. Down-weights easy examples."""

    def __init__(self, gamma: float = 2.0, alpha: float | None = None, reduction: str = "mean"):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = functional.cross_entropy(logits, targets.long(), reduction="none")
        pt = torch.exp(-ce)
        focal = (1 - pt) ** self.gamma * ce
        if self.alpha is not None:
            alpha_t = self.alpha[targets.long()]
            focal = alpha_t * focal
        if self.reduction == "mean":
            return focal.mean()
        if self.reduction == "sum":
            return focal.sum()
        return focal


class LabelSmoothingCrossEntropy(nn.Module):
    """Cross-entropy with label smoothing. Helps with overconfident predictions on rare bins."""

    def __init__(self, smoothing: float = 0.1, reduction: str = "mean"):
        super().__init__()
        self.smoothing = smoothing
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        n_classes = logits.size(-1)
        log_probs = functional.log_softmax(logits, dim=-1)
        with torch.no_grad():
            smooth_targets = torch.zeros_like(log_probs).fill_(self.smoothing / (n_classes - 1))
            smooth_targets.scatter_(1, targets.unsqueeze(1).long(), 1.0 - self.smoothing)
        loss = (-smooth_targets * log_probs).sum(dim=-1)
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss
