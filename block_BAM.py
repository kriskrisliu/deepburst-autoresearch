import torch
import torch.nn as nn
import torch.nn.functional as F

class BurstAwareBAM(nn.Module):
    def __init__(self, channels, reduction=16, dilation=4):
        super().__init__()

        # ===== Burst Channel Attention =====
        self.ca_burst = nn.Sequential(
            nn.Conv2d(2 * channels, channels // reduction, 1, bias=False),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False)
        )

        # ===== Burst Spatial Attention =====
        self.sa_burst = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=3, padding=dilation, dilation=dilation, bias=False)
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x shape: [N, C, H, W] where N can be:
        #   - burst_size (when batch_size=1, original design)
        #   - batch_size * burst_size (when batch_size > 1)
        
        # ===== Burst statistics (vectorized for both batch_size=1 and >1) =====
        # Assume N is always divisible by burst_size in this model
        burst_size = 8  # should match args.burst
        N, C, H, W = x.shape
        batch_size = N // burst_size

        # [B, burst, C, H, W]
        x_reshaped = x.view(batch_size, burst_size, C, H, W)

        # x_mean/x_var: [B, 1, C, H, W]
        x_mean = x_reshaped.mean(dim=1, keepdim=True)
        x_var = x_reshaped.var(dim=1, keepdim=True)

        # ===== Channel Attention =====
        gap_mean = x_mean.mean(dim=(3, 4), keepdim=True).squeeze(1)  # [B, C, 1, 1]
        gap_var  = x_var.mean(dim=(3, 4), keepdim=True).squeeze(1)   # [B, C, 1, 1]
        ca_input = torch.cat([gap_mean, gap_var], dim=1)             # [B, 2C, 1, 1]
        Mc = self.ca_burst(ca_input)  # [B, C, 1, 1]

        # ===== Spatial Attention =====
        sm = x_mean.mean(dim=2, keepdim=True).squeeze(1)  # [B, 1, H, W]
        sv = x_var.mean(dim=2, keepdim=True).squeeze(1)   # [B, 1, H, W]
        sa_input = torch.cat([sm, sv], dim=1)             # [B, 2, H, W]
        Ms = self.sa_burst(sa_input)  # [B, 1, H, W]

        # ===== Fusion =====
        att = self.sigmoid(Mc + Ms)  # [B, C, H, W] via broadcasting

        # Broadcast attention across burst dimension and restore original shape
        x_out = x_reshaped * att.unsqueeze(1)  # [B, burst, C, H, W]
        return x_out.view(N, C, H, W)
