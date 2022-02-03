"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_msssim import ms_ssim


class SSIMLoss(nn.Module):
    """
    SSIM loss module.
    """

    def __init__(self, win_size: int = 7, k1: float = 0.01, k2: float = 0.03):
        """
        Args:
            win_size: Window size for SSIM calculation.
            k1: k1 parameter for SSIM calculation.
            k2: k2 parameter for SSIM calculation.
        """
        super().__init__()
        self.win_size = win_size
        self.k1, self.k2 = k1, k2
        self.register_buffer("w", torch.ones(1, 1, win_size, win_size) / win_size ** 2)
        NP = win_size ** 2
        self.cov_norm = NP / (NP - 1)

    def forward(self, X: torch.Tensor, Y: torch.Tensor, data_range: torch.Tensor):
        assert isinstance(self.w, torch.Tensor)

        data_range = data_range[:, None, None, None]
        C1 = (self.k1 * data_range) ** 2
        C2 = (self.k2 * data_range) ** 2
        ux = F.conv2d(X, self.w)  # typing: ignore
        uy = F.conv2d(Y, self.w)  #
        uxx = F.conv2d(X * X, self.w)
        uyy = F.conv2d(Y * Y, self.w)
        uxy = F.conv2d(X * Y, self.w)
        vx = self.cov_norm * (uxx - ux * ux)
        vy = self.cov_norm * (uyy - uy * uy)
        vxy = self.cov_norm * (uxy - ux * uy)
        A1, A2, B1, B2 = (
            2 * ux * uy + C1,
            2 * vxy + C2,
            ux ** 2 + uy ** 2 + C1,
            vx + vy + C2,
        )
        D = B1 * B2
        S = (A1 * A2) / D

        return 1 - S.mean()

class MS_SSIMLoss(nn.Module):
    def __init__(self, win_size=11):
        super().__init__()
        self.win_size = win_size

    def forward(self, X: torch.Tensor, Y: torch.Tensor, data_range: torch.Tensor):
        # restrict X and Y to between 0 and 1 (approximately)
        X_normed = (X + 6) / 12
        Y_normed = (Y + 6) / 12
        return 1 - ms_ssim(X_normed, Y_normed, data_range=1, size_average=False, win_size=self.win_size)

class CombinationLoss(nn.Module):
    def __init__(self, simple_loss, complex_loss, alpha):
        super().__init__()
        self.simple_loss = simple_loss
        self.complex_loss = complex_loss
        self.alpha = alpha

    def forward(self, X: torch.Tensor, Y: torch.Tensor, data_range: torch.Tensor):
        return self.alpha * self.simple_loss(X, Y) + (1 - self.alpha) * self.complex_loss.forward(X, Y, data_range)