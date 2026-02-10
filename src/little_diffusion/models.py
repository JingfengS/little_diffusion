import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class BabyUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, dim=64):
        super().__init__()
        self.time_mlp = nn.Sequential(nn.Linear(1, dim), nn.GELU(), nn.Linear(dim, dim))

        # Encoder
        self.inc = nn.Sequential(
            nn.Conv2d(in_channels, dim, kernel_size=3, padding=1),
            nn.GroupNorm(8, dim),
            nn.GELU(),
        )
        self.downs1 = nn.Sequential(
            nn.Conv2d(dim, dim * 2, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, dim * 2),
            nn.GELU(),
        )
        self.downs2 = nn.Sequential(
            nn.Conv2d(dim * 2, dim * 4, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, dim * 4),
            nn.GELU(),
        )
        self.downs3 = nn.Sequential(
            nn.Conv2d(dim * 4, dim * 8, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, dim * 8),
            nn.GELU(),
        )

        # Decoder
        self.up3 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.outc3 = nn.Sequential(
            nn.Conv2d(dim * 8 + dim * 4, dim * 4, 3, padding=1),
            nn.GroupNorm(8, dim * 4),
            nn.GELU(),
        )
        self.up2 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.outc2 = nn.Sequential(
            nn.Conv2d(dim * 4 + dim * 2, dim * 2, 3, padding=1),
            nn.GroupNorm(8, dim * 2),
            nn.GELU(),
        )
        self.up1 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.outc1 = nn.Sequential(
            nn.Conv2d(dim * 2 + dim, dim, 3, padding=1), nn.GroupNorm(8, dim), nn.GELU()
        )
        self.outc = nn.Conv2d(dim, out_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, in_channels, height, width)
            t: (batch_size, 1)
        Returns:
            (batch_size, out_channels, height, width)
        """
        t = t.to(x.device)
        t_emb = self.time_mlp(t)[..., None, None] # (batch_size, dim, 1, 1)

        x1 = self.inc(x)
        x2 = self.downs1(x1 + t_emb)
        x3 = self.downs2(x2)
        x4 = self.downs3(x3)

        x_up3 = self.up3(x4)
        if x_up3.shape[2:] != x3.shape[2:]:
            x3 = F.interpolate(x3, size=x_up3.shape[2:], mode="bilinear", align_corners=True)

        x_out3 = self.outc3(torch.cat([x_up3, x3], dim=1))
        x_up2 = self.up2(x_out3)
        if x_up2.shape[2:] != x2.shape[2:]:
            x2 = F.interpolate(x2, size=x_up2.shape[2:], mode="bilinear", align_corners=True)
        x_out2 = self.outc2(torch.cat([x_up2, x2], dim=1))
        x_up1 = self.up1(x_out2)
        if x_up1.shape[2:] != x1.shape[2:]:
            x1 = F.interpolate(x1, size=x_up1.shape[2:], mode="bilinear", align_corners=True)
        x_out1 = self.outc1(torch.cat([x_up1, x1], dim=1))
        x_out = self.outc(x_out1)
        return x_out


