import torch
import torch.nn as nn

class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1, bias=False),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, kernel_size=1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        
        avg_pool = self.avg_pool(x)
        max_pool = self.max_pool(x)
        
        avg_out = self.fc(avg_pool)
        max_out = self.fc(max_pool)
        
        scale = avg_out + max_out
        
        return x * scale


class SpatialAttention(nn.Module):
    def __init__(self):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(4, 1, kernel_size=7, padding=3, groups=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(1, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        mean_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        min_out, _ = torch.min(x, dim=1, keepdim=True)
        sum_out = torch.sum(x, dim=1, keepdim=True)
        
        pool = torch.cat([mean_out, max_out, min_out, sum_out], dim=1)
        # pool = torch.cat([mean_out, max_out], dim=1)
        attention = self.conv(pool)
        
        return x * attention

class FSCB(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(FSCB, self).__init__()
        
        self.channel_attention = ChannelAttention(in_channels, reduction)

        self.spatial_attention = SpatialAttention()
        
        self.fuse_conv = nn.Sequential(
            nn.Conv2d(3 * in_channels, in_channels, kernel_size=1, padding=0),
            nn.LeakyReLU(0.1, inplace=True)
        )

    def forward(self, x):
        
        fused = torch.cat([x,
                           self.spatial_attention(x),
                           self.channel_attention(x)
                           ], dim=1)

        fused = self.fuse_conv(fused)

        out = fused + x

        return out
