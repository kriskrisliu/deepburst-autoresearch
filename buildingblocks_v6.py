import torch
from torch import nn as nn
from torch.nn import functional as F
from block_FSCB import FSCB
from block_BAM import BurstAwareBAM

class ConvBlock(nn.Sequential):
    def __init__(self, in_channels, out_channels, encoder=True, kernel_size=3):
        super(ConvBlock, self).__init__()

        if encoder:
            # we're in the encoder path
            conv1_in_channels = in_channels
            conv1_out_channels = out_channels // 2
            if conv1_out_channels < in_channels:
                conv1_out_channels = in_channels
            conv2_in_channels, conv2_out_channels = conv1_out_channels, out_channels
        else:
            # we're in the decoder path, decrease the number of channels in the 1st convolution
            conv1_in_channels, conv1_out_channels = in_channels, out_channels
            conv2_in_channels, conv2_out_channels = out_channels, out_channels

        # 将层直接添加到 nn.Sequential 中
        self.add_module("conv1", nn.Conv2d(conv1_in_channels, conv1_out_channels, kernel_size, padding=1))
        self.add_module("relu1", nn.LeakyReLU(negative_slope=0.1, inplace=True))
        self.add_module("conv2", nn.Conv2d(conv2_in_channels, conv2_out_channels, kernel_size, padding=1))
        self.add_module("relu2", nn.LeakyReLU(negative_slope=0.1, inplace=True))


class Encoder(nn.Module):
    def __init__(self, in_channels, out_channels, conv_kernel_size=3, apply_pooling=True, pool_type='max', if_BAM=True, if_FSCB=True):
        super(Encoder, self).__init__()
        assert pool_type in ['max', 'avg']
        if apply_pooling:
            if pool_type == 'max':
                self.pooling = nn.MaxPool2d(kernel_size=2)
            else:
                self.pooling = nn.AvgPool2d(kernel_size=2)
        else:
            self.pooling = None
            
        self.ConvBlock = ConvBlock(in_channels, out_channels, encoder=True, kernel_size=conv_kernel_size)
        
        # Always create these attributes for TorchScript compatibility
        # Use Identity module when not needed
        if if_BAM:
            self.BurstAwareBAM = BurstAwareBAM(out_channels)
        else:
            self.BurstAwareBAM = nn.Identity()
        
        if if_FSCB:
            self.FSCAttention = FSCB(out_channels, reduction=16)
        else:
            self.FSCAttention = nn.Identity()
            
        self.if_BAM = if_BAM
        self.if_FSCB = if_FSCB
        
    def forward(self, x):
        if self.pooling is not None:
            x = self.pooling(x)
            
        x = self.ConvBlock(x)
        
        # Always call these modules (Identity when not needed)
        # This ensures TorchScript compatibility
        x = self.BurstAwareBAM(x)
        x = self.FSCAttention(x)
        
        return x
    

class Decoder(nn.Module):
    def __init__(self, in_channels, out_channels, conv_kernel_size=3):
        super(Decoder, self).__init__()
        
        self.ConvBlock = ConvBlock(in_channels, out_channels, encoder=False, kernel_size=conv_kernel_size)

    def forward(self, x, encoder_features):
        output_size = encoder_features.size()[2:]
        x = F.interpolate(x, size=output_size, mode='nearest')

        x = torch.cat((encoder_features, x), dim=1)

        x = self.ConvBlock(x)

        return x
    
    
class Fusion(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, burst_size=8):
        super(Fusion, self).__init__()
        self.burst_size = burst_size
        
        self.fusion_conv = ConvBlock(in_channels, out_channels, encoder=False, kernel_size=kernel_size)

    def forward(self, x):
        # x: [B*burst, C, H, W] or [burst, C, H, W]
        B_mul, C, H, W = x.shape

        # Unified implementation for both batch_size=1 and >1 (avoids Python condition on tensor)
        B = B_mul // self.burst_size

        # [B, burst, C, H, W]
        x = x.view(B, self.burst_size, C, H, W)

        # 👉 关键：把 batch 和 channel 合并，一次性做 conv
        # [B*C, burst, H, W]
        x = x.permute(0, 2, 1, 3, 4).reshape(B * C, self.burst_size, H, W)

        x = self.fusion_conv(x)      # [B*C, 1, H, W]

        # reshape 回来
        x = x.view(B, C, H, W)
        return x

    
    def forward_old(self, x):
        # x shape: [N, channels, H, W] where N can be:
        #   - burst_size (when batch_size=1, original design)
        #   - batch_size * burst_size (when batch_size > 1)
        
        # Original Fusion layer: just transpose, conv, transpose (no mean!)
        # Input: [burst_size, channels, H, W]
        # -> transpose: [channels, burst_size, H, W]
        # -> fusion_conv: [channels, out_channels=1, H, W]
        # -> transpose: [out_channels=1, channels, H, W] = [1, channels, H, W]
        # Output: [1, channels, H, W] (NOT [1, 1, H, W]!)
        
        # Handle batch_size > 1 case: process each batch sample identically to batch_size=1
        if x.shape[0] > self.burst_size and x.shape[0] % self.burst_size == 0:
            # batch_size > 1: split into batches and process each exactly like batch_size=1
            batch_size = x.shape[0] // self.burst_size
            channels, h, w = x.shape[1], x.shape[2], x.shape[3]
            
            # Reshape to [batch_size, burst_size, channels, H, W]
            x = x.view(batch_size, self.burst_size, channels, h, w)
            
            # Process each batch sample identically to the batch_size=1 case
            outputs = []
            for b in range(batch_size):
                batch_x = x[b]  # [burst_size, channels, H, W] - same as batch_size=1 input
                # Apply EXACT same operations as batch_size=1 case (no mean!)
                batch_x = batch_x.transpose(0, 1).contiguous()  # [channels, burst_size, H, W]
                batch_x = self.fusion_conv(batch_x)  # [channels, out_channels=1, H, W]
                batch_x = batch_x.transpose(0, 1).contiguous()  # [out_channels=1, channels, H, W] = [1, channels, H, W]
                outputs.append(batch_x)  # [1, channels, H, W]
            
            # Concatenate along batch dimension: [batch_size, channels, H, W]
            x = torch.cat(outputs, dim=0)  # [batch_size, channels, H, W]
        else:
            # Original case: batch_size = 1, first dim is burst_size
            # Input: [burst_size, channels, H, W]
            x = x.transpose(0, 1).contiguous()  # [channels, burst_size, H, W]
            x = self.fusion_conv(x)  # [channels, out_channels=1, H, W]
            x = x.transpose(0, 1).contiguous()  # [out_channels=1, channels, H, W] = [1, channels, H, W]
        
        return x