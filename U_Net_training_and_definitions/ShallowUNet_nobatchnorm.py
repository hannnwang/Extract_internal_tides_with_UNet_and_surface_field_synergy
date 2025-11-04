from unet_parts import *
from torch.utils.checkpoint import checkpoint
class Up_for_shallowUNets(nn.Module):
    """Upscaling then double conv"""
    def __init__(self, in_channels, out_channels, bilinear=True):
            super().__init__()
            if bilinear:
                self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            else:
                self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)
    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

#Receptive field
#Orignial U-Net with 4 layers: 200*200 patch in the input affects one pixel in the output. 
#Receptive field: 44-by-44 patch in the input affects one pixel in the output. 
class TwolayerUNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False, Nbase = 16):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        #self.inpBN  = nn.BatchNorm2d(n_channels) 
        self.inc = (DoubleConv(n_channels, Nbase))
        self.down1 = (Down(Nbase, Nbase*2))
        self.down2 = (Down(Nbase*2, Nbase*4))
        #self.down3 = (Down(Nbase*4, Nbase*8))
        #self.down4 = (Down(Nbase*8, Nbase*16 // factor))
        #self.up1 = (Up(Nbase*16, Nbase*8 // factor, bilinear))
        #self.up2 = (Up(Nbase*8, Nbase*4 // factor, bilinear))
        self.up3 = (Up_for_shallowUNets(Nbase*4 + Nbase*2, Nbase*2, bilinear))
        self.up4 = (Up_for_shallowUNets(Nbase*2 + Nbase, Nbase, bilinear))
        self.outc = (OutConv(Nbase, n_classes))


    def forward(self, x):
        x1 = checkpoint(self.inc, x, use_reentrant=False)
        x2 = checkpoint(self.down1, x1, use_reentrant=False)
        x3 = checkpoint(self.down2, x2, use_reentrant=False)
        x = checkpoint(self.up3, x3, x2, use_reentrant=False)
        x = checkpoint(self.up4, x, x1, use_reentrant=False)
        logits = checkpoint(self.outc, x, use_reentrant=False)
        return logits

#Recptive field: 18*18
#Turns out that the memory blows up too fast.
class OnelayerUNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False, Nbase=16):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.inc = DoubleConv(n_channels, Nbase)
        self.down1 = Down(Nbase, Nbase*2)
        # Only one down and one up!
        self.up1 = Up_for_shallowUNets(Nbase*2 + Nbase, Nbase, bilinear)
        self.outc = OutConv(Nbase, n_classes)

    def forward(self, x):
        x1 = checkpoint(self.inc, x, use_reentrant=False)
        x2 = checkpoint(self.down1, x1, use_reentrant=False)
        x = checkpoint(self.up1, x2, x1, use_reentrant=False)
        logits = checkpoint(self.outc, x, use_reentrant=False)
        return logits


