# coding:utf-8
#anaconda/anaconda3/envs/python37
'''
@File    :   unet.py
@Time    :   2023/08/07 20:04:01
@Author  :   Asuka 
@Contact :   shixulei@whu.edu.cn
@Desc    :   None
'''

""" Full assembly of the parts to form the complete network """

from .unet_parts import *
from torchsummary import summary


class WNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(WNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        #4*128*128
        self.inc = DoubleConv(3, 32)#32*128*128
        self.down1 = Down(32, 64)#64*64*64
        self.down2 = Down(64, 128)#128*32*32
        self.down3 = Down(128, 256)#256*16*16
        self.down4 = Down(256, 512)#512*8*8
        self.up = Up(1024, 512, bilinear)#512*16*16
        self.up1 = Up(512, 256, bilinear)#256*16*16
        self.up2 = Up(256, 128, bilinear)#128*32*32
        self.up3 = Up(128, 64, bilinear)#64*64*64
        self.up4 = Up(64, 32, bilinear)#32*128*128
        self.outc = OutConv(64, n_classes)#3*128*128

    def forward(self, x):
        x13 = x[:, :3, :, :]  # First three channels #3*128*128
        y24 = x[:, 1:4, :, :]  # Last three channels #3*128*128
        
        #down
        x1 = self.inc(x13)#32*128*128
        x2 = self.down1(x1)#64*64*64
        x3 = self.down2(x2)#128*32*32
        x4 = self.down3(x3)#256*16*16
        x5 = self.down4(x4)#512*8*8
    
        y1 = self.inc(y24)#32*128*128
        y2 = self.down1(y1)#64*64*64
        y3 = self.down2(y2)#128*32*32
        y4 = self.down3(y3)#256*16*16
        y5 = self.down4(y4)#512*8*8
        
        #skip connection
        xy5 = torch.cat([x5, y5], dim=1)#1024*8*8
        
        #up1
        x = self.up1(x5, x4)#256*16*16
        y = self.up1(y5, y4)#256*16*16
        xy4 = torch.cat([x, y], dim=1)#512*16*16
        xy = self.up(xy5, xy4)#512*16*16
        #up2
        x = self.up2(x, x3)#128*32*32
        y = self.up2(y, y3)#128*32*32
        xy3 = torch.cat([x, y], dim=1)#256*32*32
        xy = self.up1(xy, xy3)#256*32*32
        #up3
        x = self.up3(x, x2)#64*64*64
        y = self.up3(y, y2)#64*64*64
        xy2 = torch.cat([x, y], dim=1)#128*64*64
        xy = self.up2(xy, xy2)#128*64*64
        #up4
        x = self.up4(x, x1)#32*128*128
        y = self.up4(y, y1)#32*128*128
        xy1 = torch.cat([x, y], dim=1)#64*128*128
        xy = self.up3(xy, xy1)#64*128*128
 
        logits = self.outc(xy)#4*128*128
        return logits

if __name__ == '__main__':
    unet = WNet(3, 3).cpu()
    summary(unet, input_size=(4, 128, 128), device='cpu')