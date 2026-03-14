import torch.nn as nn
from utils import create_feature_maps
from buildingblocks_v6 import Encoder, Decoder, Fusion


class DeepBurst(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, f_maps=64, number_of_fmaps=6, burst_size=8):
        super(DeepBurst, self).__init__()
        
        if isinstance(f_maps, int):
            f_maps = create_feature_maps(f_maps, number_of_fmaps)

        self.encoder1 = Encoder(in_channels, f_maps[0], apply_pooling=False, if_BAM=False, if_FSCB=False)
        self.encoder2 = Encoder(f_maps[0], f_maps[1], apply_pooling=True, if_BAM=False, if_FSCB=False)
        self.encoder3 = Encoder(f_maps[1], f_maps[2], apply_pooling=True, if_BAM=False, if_FSCB=True)
        self.encoder4 = Encoder(f_maps[2], f_maps[3], apply_pooling=True, if_BAM=True, if_FSCB=False)
        self.encoder5 = Encoder(f_maps[3], f_maps[4], apply_pooling=True, if_BAM=False, if_FSCB=True)
        self.encoder6 = Encoder(f_maps[4], f_maps[4], apply_pooling=True, if_BAM=True, if_FSCB=True)
        
        self.fusion = Fusion(in_channels=burst_size, out_channels=1, burst_size=burst_size)
        
        self.decoder6 = Decoder(f_maps[4] + f_maps[4], f_maps[4])
        self.decoder5 = Decoder(f_maps[4] + f_maps[3], f_maps[3])
        self.decoder4 = Decoder(f_maps[3] + f_maps[2], f_maps[2])
        self.decoder3 = Decoder(f_maps[2] + f_maps[1], f_maps[1])
        self.decoder2 = Decoder(f_maps[1] + f_maps[0], f_maps[0])
        # self.decoder1 = Decoder(f_maps[0], out_channels)

        self.final_conv = nn.Conv2d(f_maps[0], out_channels, kernel_size=1)

    def forward(self, x):
        
        # x = x.view(-1, *x.shape[-2:])
        # x = x.unsqueeze(1)

        enc1 = self.encoder1(x)
        enc2 = self.encoder2(enc1)
        enc3 = self.encoder3(enc2)
        enc4 = self.encoder4(enc3)
        enc5 = self.encoder5(enc4)
        enc6 = self.encoder6(enc5)
        
        # Handle mean operation over burst dimension for both batch_size=1 and >1
        # Assume first dimension is always batch_size * burst_size
        B_mul = enc1.shape[0]
        burst = self.fusion.burst_size
        batch_size = B_mul // burst

        # Reshape to [batch_size, burst_size, ...] and take mean over burst dimension
        enc1_m = enc1.view(batch_size, burst, *enc1.shape[1:]).mean(dim=1)
        enc2_m = enc2.view(batch_size, burst, *enc2.shape[1:]).mean(dim=1)
        enc3_m = enc3.view(batch_size, burst, *enc3.shape[1:]).mean(dim=1)
        enc4_m = enc4.view(batch_size, burst, *enc4.shape[1:]).mean(dim=1)
        enc5_m = enc5.view(batch_size, burst, *enc5.shape[1:]).mean(dim=1)
        
        enc6 = self.fusion(enc6)
        # enc6 = enc6.mean(dim=0, keepdim=True)

        dec = self.decoder6(enc6, enc5_m)
        dec = self.decoder5(dec, enc4_m)
        dec = self.decoder4(dec, enc3_m)
        dec = self.decoder3(dec, enc2_m)
        dec = self.decoder2(dec, enc1_m)
        # dec3 = self.decoder3(dec4)
        # dec2 = self.decoder2(dec3)
        # dec1 = self.decoder1(dec2 + enc1.mean(dim=0, keepdim=True))
            
        out = self.final_conv(dec)
        # out = out.unsqueeze(0)
                 
        return out