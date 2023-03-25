import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import matplotlib.pyplot as plt
class REBNCONV(nn.Module):
    def __init__(self,in_ch=3,out_ch=3,dirate=1):
        super(REBNCONV,self).__init__()
        self.conv_s1 = nn.Conv2d(in_ch,out_ch,3,padding=1*dirate,dilation=1*dirate)
        self.bn_s1 = nn.BatchNorm2d(out_ch)
        self.relu_s1 = nn.ReLU(inplace=True)
    def forward(self,x):
        hx = x
        xout = self.relu_s1(self.bn_s1(self.conv_s1(hx)))
        return xout

## upsample tensor 'src' to have the same spatial size with tensor 'tar'
def _upsample_like(src,tar):
    src = F.upsample(src,size=tar.shape[2:],mode='bilinear')
    return src


### RSU-7 ###
class RSU7(nn.Module):#UNet07DRES(nn.Module):
    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(RSU7,self).__init__()
        self.rebnconvin = REBNCONV(in_ch,out_ch,dirate=1)
        self.rebnconv1 = REBNCONV(out_ch,mid_ch,dirate=1)
        self.pool1 = nn.MaxPool2d(2,stride=2,ceil_mode=True)
        self.rebnconv2 = REBNCONV(mid_ch,mid_ch,dirate=1)
        self.pool2 = nn.MaxPool2d(2,stride=2,ceil_mode=True)
        self.rebnconv3 = REBNCONV(mid_ch,mid_ch,dirate=1)
        self.pool3 = nn.MaxPool2d(2,stride=2,ceil_mode=True)
        self.rebnconv4 = REBNCONV(mid_ch,mid_ch,dirate=1)
        self.pool4 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.rebnconv5 = REBNCONV(mid_ch,mid_ch,dirate=1)
        self.pool5 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.rebnconv6 = REBNCONV(mid_ch,mid_ch,dirate=1)

        self.rebnconv7 = REBNCONV(mid_ch,mid_ch,dirate=2)

        self.rebnconv6d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
        self.rebnconv5d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
        self.rebnconv4d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
        self.rebnconv3d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
        self.rebnconv2d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
        self.rebnconv1d = REBNCONV(mid_ch*2,out_ch,dirate=1)

    def forward(self,x):

        hx = x
        hxin = self.rebnconvin(hx)

        hx1 = self.rebnconv1(hxin)
        hx = self.pool1(hx1)

        hx2 = self.rebnconv2(hx)
        hx = self.pool2(hx2)

        hx3 = self.rebnconv3(hx)
        hx = self.pool3(hx3)

        hx4 = self.rebnconv4(hx)
        hx = self.pool4(hx4)

        hx5 = self.rebnconv5(hx)
        hx = self.pool5(hx5)

        hx6 = self.rebnconv6(hx)

        hx7 = self.rebnconv7(hx6)

        hx6d =  self.rebnconv6d(torch.cat((hx7,hx6),1))
        hx6dup = _upsample_like(hx6d,hx5)

        hx5d =  self.rebnconv5d(torch.cat((hx6dup,hx5),1))
        hx5dup = _upsample_like(hx5d,hx4)

        hx4d = self.rebnconv4d(torch.cat((hx5dup,hx4),1))
        hx4dup = _upsample_like(hx4d,hx3)

        hx3d = self.rebnconv3d(torch.cat((hx4dup,hx3),1))
        hx3dup = _upsample_like(hx3d,hx2)

        hx2d = self.rebnconv2d(torch.cat((hx3dup,hx2),1))
        hx2dup = _upsample_like(hx2d,hx1)

        hx1d = self.rebnconv1d(torch.cat((hx2dup,hx1),1))

        return hx1d + hxin


### RSU-6 ###
class RSU6(nn.Module):#UNet06DRES(nn.Module):

    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(RSU6,self).__init__()

        self.rebnconvin = REBNCONV(in_ch,out_ch,dirate=1)

        self.rebnconv1 = REBNCONV(out_ch,mid_ch,dirate=1)
        self.pool1 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.rebnconv2 = REBNCONV(mid_ch,mid_ch,dirate=1)
        self.pool2 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.rebnconv3 = REBNCONV(mid_ch,mid_ch,dirate=1)
        self.pool3 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.rebnconv4 = REBNCONV(mid_ch,mid_ch,dirate=1)
        self.pool4 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.rebnconv5 = REBNCONV(mid_ch,mid_ch,dirate=1)

        self.rebnconv6 = REBNCONV(mid_ch,mid_ch,dirate=2)

        self.rebnconv5d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
        self.rebnconv4d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
        self.rebnconv3d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
        self.rebnconv2d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
        self.rebnconv1d = REBNCONV(mid_ch*2,out_ch,dirate=1)

    def forward(self,x):

        hx = x

        hxin = self.rebnconvin(hx)

        hx1 = self.rebnconv1(hxin)
        hx = self.pool1(hx1)

        hx2 = self.rebnconv2(hx)
        hx = self.pool2(hx2)

        hx3 = self.rebnconv3(hx)
        hx = self.pool3(hx3)

        hx4 = self.rebnconv4(hx)
        hx = self.pool4(hx4)

        hx5 = self.rebnconv5(hx)

        hx6 = self.rebnconv6(hx5)


        hx5d =  self.rebnconv5d(torch.cat((hx6,hx5),1))
        hx5dup = _upsample_like(hx5d,hx4)

        hx4d = self.rebnconv4d(torch.cat((hx5dup,hx4),1))
        hx4dup = _upsample_like(hx4d,hx3)

        hx3d = self.rebnconv3d(torch.cat((hx4dup,hx3),1))
        hx3dup = _upsample_like(hx3d,hx2)

        hx2d = self.rebnconv2d(torch.cat((hx3dup,hx2),1))
        hx2dup = _upsample_like(hx2d,hx1)

        hx1d = self.rebnconv1d(torch.cat((hx2dup,hx1),1))

        return hx1d + hxin


### RSU-5 ###
class RSU5(nn.Module):#UNet05DRES(nn.Module):

    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(RSU5,self).__init__()

        self.rebnconvin = REBNCONV(in_ch,out_ch,dirate=1)

        self.rebnconv1 = REBNCONV(out_ch,mid_ch,dirate=1)
        self.pool1 = nn.AvgPool2d(2,stride=2,ceil_mode=True)

        self.rebnconv2 = REBNCONV(mid_ch,mid_ch,dirate=1)
        self.pool2 = nn.AvgPool2d(2,stride=2,ceil_mode=True)

        self.rebnconv3 = REBNCONV(mid_ch,mid_ch,dirate=1)
        self.pool3 = nn.AvgPool2d(2,stride=2,ceil_mode=True)

        self.rebnconv4 = REBNCONV(mid_ch,mid_ch,dirate=1)

        self.rebnconv5 = REBNCONV(mid_ch,mid_ch,dirate=2)

        self.rebnconv4d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
        self.rebnconv3d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
        self.rebnconv2d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
        self.rebnconv1d = REBNCONV(mid_ch*2,out_ch,dirate=1)

    def forward(self,x):

        hx = x

        hxin = self.rebnconvin(hx)

        hx1 = self.rebnconv1(hxin)
        hx = self.pool1(hx1)

        hx2 = self.rebnconv2(hx)
        hx = self.pool2(hx2)

        hx3 = self.rebnconv3(hx)
        hx = self.pool3(hx3)

        hx4 = self.rebnconv4(hx)

        hx5 = self.rebnconv5(hx4)

        hx4d = self.rebnconv4d(torch.cat((hx5,hx4),1))
        hx4dup = _upsample_like(hx4d,hx3)

        hx3d = self.rebnconv3d(torch.cat((hx4dup,hx3),1))
        hx3dup = _upsample_like(hx3d,hx2)

        hx2d = self.rebnconv2d(torch.cat((hx3dup,hx2),1))
        hx2dup = _upsample_like(hx2d,hx1)

        hx1d = self.rebnconv1d(torch.cat((hx2dup,hx1),1))

        return hx1d + hxin

### RSU-4 ###
class RSU4(nn.Module):#UNet04DRES(nn.Module):

    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(RSU4,self).__init__()

        self.rebnconvin = REBNCONV(in_ch,out_ch,dirate=1)

        self.rebnconv1 = REBNCONV(out_ch,mid_ch,dirate=1)
        self.pool1 = nn.AvgPool2d(2,stride=2,ceil_mode=True)

        self.rebnconv2 = REBNCONV(mid_ch,mid_ch,dirate=1)
        self.pool2 = nn.AvgPool2d(2,stride=2,ceil_mode=True)

        self.rebnconv3 = REBNCONV(mid_ch,mid_ch,dirate=1)

        self.rebnconv4 = REBNCONV(mid_ch,mid_ch,dirate=2)

        self.rebnconv3d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
        self.rebnconv2d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
        self.rebnconv1d = REBNCONV(mid_ch*2,out_ch,dirate=1)

    def forward(self,x):

        hx = x

        hxin = self.rebnconvin(hx)

        hx1 = self.rebnconv1(hxin)
        hx = self.pool1(hx1)

        hx2 = self.rebnconv2(hx)
        hx = self.pool2(hx2)

        hx3 = self.rebnconv3(hx)

        hx4 = self.rebnconv4(hx3)

        hx3d = self.rebnconv3d(torch.cat((hx4,hx3),1))
        hx3dup = _upsample_like(hx3d,hx2)

        hx2d = self.rebnconv2d(torch.cat((hx3dup,hx2),1))
        hx2dup = _upsample_like(hx2d,hx1)

        hx1d = self.rebnconv1d(torch.cat((hx2dup,hx1),1))

        return hx1d + hxin

### RSU-4F ###
class RSU4F(nn.Module):#UNet04FRES(nn.Module):

    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(RSU4F,self).__init__()

        self.rebnconvin = REBNCONV(in_ch,out_ch,dirate=1)

        self.rebnconv1 = REBNCONV(out_ch,mid_ch,dirate=1)
        self.rebnconv2 = REBNCONV(mid_ch,mid_ch,dirate=2)
        self.rebnconv3 = REBNCONV(mid_ch,mid_ch,dirate=4)

        self.rebnconv4 = REBNCONV(mid_ch,mid_ch,dirate=8)

        self.rebnconv3d = REBNCONV(mid_ch*2,mid_ch,dirate=4)
        self.rebnconv2d = REBNCONV(mid_ch*2,mid_ch,dirate=2)
        self.rebnconv1d = REBNCONV(mid_ch*2,out_ch,dirate=1)

    def forward(self,x):

        hx = x

        hxin = self.rebnconvin(hx)

        hx1 = self.rebnconv1(hxin)
        hx2 = self.rebnconv2(hx1)
        hx3 = self.rebnconv3(hx2)

        hx4 = self.rebnconv4(hx3)

        hx3d = self.rebnconv3d(torch.cat((hx4,hx3),1))
        hx2d = self.rebnconv2d(torch.cat((hx3d,hx2),1))
        hx1d = self.rebnconv1d(torch.cat((hx2d,hx1),1))

        return hx1d + hxin
#相同尺寸，面阵
class depthwise_separable_conv(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(depthwise_separable_conv, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=stride, padding=1,
                                   groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out
class UpConv(nn.Module):
    def __init__(self, inc, outc, scale=2):
        super(UpConv, self).__init__()
        self.scale = scale
        self.conv = nn.Conv2d(inc, outc, 3, stride=1, padding=1)

    def forward(self, x):
        return self.conv(F.interpolate(x, scale_factor=self.scale, mode='bilinear', align_corners=True))
class CEBlock(nn.Module):
    def __init__(self, in_channels=16, out_channels=16):
        super(CEBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.gap = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.BatchNorm2d(self.in_channels)
        )

        self.conv_gap = nn.Sequential(
            nn.Conv2d(self.in_channels, self.out_channels, 1, stride=1, padding=0),
            nn.BatchNorm2d(self.out_channels),
            nn.ReLU()
        )

        # Note: in paper here is naive conv2d, no bn-relu
        self.conv_last = nn.Conv2d(
            in_channels=self.out_channels,
            out_channels=self.out_channels,
            kernel_size=3,
            stride=1,
            padding=1)

    def forward(self, x):
        identity = x
        x = self.gap(x)
        x = self.conv_gap(x)
        x = identity + x
        x = self.conv_last(x)
        return x

def my_layer_norm(feat):
    mean = feat.mean((2, 3), keepdim=True)
    std = feat.std((2, 3), keepdim=True) + 1e-9
    feat = 2 * (feat - mean) / std - 1
    feat = 5 * feat
    return feat

class My_RSU7(nn.Module):#UNet07DRES(nn.Module):

    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(My_RSU7,self).__init__()

        self.rebnconvin = REBNCONV(in_ch,out_ch,dirate=1)

        self.rebnconv1 = REBNCONV(out_ch,mid_ch,dirate=1)
        self.pool1 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.rebnconv2 = REBNCONV(mid_ch,mid_ch,dirate=1)
        self.pool2 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.rebnconv3 = REBNCONV(mid_ch,mid_ch,dirate=1)
        self.pool3 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.rebnconv4 = REBNCONV(mid_ch,mid_ch,dirate=1)
        self.pool4 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.rebnconv5 = REBNCONV(mid_ch,mid_ch,dirate=1)
        self.pool5 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.rebnconv6 = REBNCONV(mid_ch,mid_ch,dirate=1)

        self.rebnconv7 = REBNCONV(mid_ch,mid_ch,dirate=2)

        self.rebnconv6d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
        self.rebnconv5d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
        self.rebnconv4d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
        self.rebnconv3d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
        self.rebnconv2d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
        self.rebnconv1d = REBNCONV(mid_ch*2,out_ch,dirate=1)

    def forward(self,x,y):

        hx = x
        hxin = self.rebnconvin(hx)

        hx1 = self.rebnconv1(hxin)
        hx = self.pool1(hx1)

        hx2 = self.rebnconv2(hx)
        hx = self.pool2(hx2)

        hx3 = self.rebnconv3(hx)
        hx = self.pool3(hx3)

        hx4 = self.rebnconv4(hx)
        hx = self.pool4(hx4)

        hx5 = self.rebnconv5(hx)
        hx = self.pool5(hx5)

        hx6 = self.rebnconv6(hx)

        hx7 = self.rebnconv7(hx6)

        hx6d =  self.rebnconv6d(torch.cat((hx7,hx6),1))
        hx6dup = _upsample_like(hx6d,hx5)

        hx5d =  self.rebnconv5d(torch.cat((hx6dup,hx5),1))
        hx5dup = _upsample_like(hx5d,hx4)

        hx4d = self.rebnconv4d(torch.cat((hx5dup,hx4),1))
        hx4dup = _upsample_like(hx4d,hx3)

        hx3d = self.rebnconv3d(torch.cat((hx4dup,hx3),1))
        hx3dup = _upsample_like(hx3d,hx2)

        hx2d = self.rebnconv2d(torch.cat((hx3dup,hx2),1))
        hx2dup = _upsample_like(hx2d,hx1)

        hx1d = self.rebnconv1d(torch.cat((hx2dup,hx1),1))

        return hx1d*y + hxin*(1-y)
### RSU-6 ###
class My_RSU6(nn.Module):#UNet06DRES(nn.Module):

    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(My_RSU6,self).__init__()

        self.rebnconvin = REBNCONV(in_ch,out_ch,dirate=1)

        self.rebnconv1 = REBNCONV(out_ch,mid_ch,dirate=1)
        self.pool1 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.rebnconv2 = REBNCONV(mid_ch,mid_ch,dirate=1)
        self.pool2 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.rebnconv3 = REBNCONV(mid_ch,mid_ch,dirate=1)
        self.pool3 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.rebnconv4 = REBNCONV(mid_ch,mid_ch,dirate=1)
        self.pool4 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.rebnconv5 = REBNCONV(mid_ch,mid_ch,dirate=1)

        self.rebnconv6 = REBNCONV(mid_ch,mid_ch,dirate=2)

        self.rebnconv5d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
        self.rebnconv4d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
        self.rebnconv3d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
        self.rebnconv2d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
        self.rebnconv1d = REBNCONV(mid_ch*2,out_ch,dirate=1)

    def forward(self,x,y):

        hx = x

        hxin = self.rebnconvin(hx)

        hx1 = self.rebnconv1(hxin)
        hx = self.pool1(hx1)

        hx2 = self.rebnconv2(hx)
        hx = self.pool2(hx2)

        hx3 = self.rebnconv3(hx)
        hx = self.pool3(hx3)

        hx4 = self.rebnconv4(hx)
        hx = self.pool4(hx4)

        hx5 = self.rebnconv5(hx)

        hx6 = self.rebnconv6(hx5)


        hx5d =  self.rebnconv5d(torch.cat((hx6,hx5),1))
        hx5dup = _upsample_like(hx5d,hx4)

        hx4d = self.rebnconv4d(torch.cat((hx5dup,hx4),1))
        hx4dup = _upsample_like(hx4d,hx3)

        hx3d = self.rebnconv3d(torch.cat((hx4dup,hx3),1))
        hx3dup = _upsample_like(hx3d,hx2)

        hx2d = self.rebnconv2d(torch.cat((hx3dup,hx2),1))
        hx2dup = _upsample_like(hx2d,hx1)

        hx1d = self.rebnconv1d(torch.cat((hx2dup,hx1),1))

        return hx1d*y + hxin*(1-y)
### My_RSU-5 ###
class My_RSU_5(nn.Module):#UNet05DRES(nn.Module):

    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(My_RSU_5,self).__init__()

        self.rebnconvin = REBNCONV(in_ch,out_ch,dirate=1)

        self.rebnconv1 = REBNCONV(out_ch,mid_ch,dirate=1)
        self.pool1 = nn.AvgPool2d(2,stride=2,ceil_mode=True)

        self.rebnconv2 = REBNCONV(mid_ch,mid_ch,dirate=1)
        self.pool2 = nn.AvgPool2d(2,stride=2,ceil_mode=True)

        self.rebnconv3 = REBNCONV(mid_ch,mid_ch,dirate=1)
        self.pool3 = nn.AvgPool2d(2,stride=2,ceil_mode=True)

        self.rebnconv4 = REBNCONV(mid_ch,mid_ch,dirate=1)

        self.rebnconv5 = REBNCONV(mid_ch,mid_ch,dirate=2)

        self.rebnconv4d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
        self.rebnconv3d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
        self.rebnconv2d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
        self.rebnconv1d = REBNCONV(mid_ch*2,out_ch,dirate=1)

    def forward(self,x, y):

        hx = x

        hxin = self.rebnconvin(hx)

        hx1 = self.rebnconv1(hxin)
        hx = self.pool1(hx1)

        hx2 = self.rebnconv2(hx)
        hx = self.pool2(hx2)

        hx3 = self.rebnconv3(hx)
        hx = self.pool3(hx3)

        hx4 = self.rebnconv4(hx)

        hx5 = self.rebnconv5(hx4)

        hx4d = self.rebnconv4d(torch.cat((hx5,hx4),1))
        hx4dup = _upsample_like(hx4d,hx3)

        hx3d = self.rebnconv3d(torch.cat((hx4dup,hx3),1))
        hx3dup = _upsample_like(hx3d,hx2)

        hx2d = self.rebnconv2d(torch.cat((hx3dup,hx2),1))
        hx2dup = _upsample_like(hx2d,hx1)

        hx1d = self.rebnconv1d(torch.cat((hx2dup,hx1),1))

        return hx1d*y + hxin*(1-y)
### My_RSU-4 ###
class My_RSU_4(nn.Module):#UNet04DRES(nn.Module):

    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(My_RSU_4,self).__init__()

        self.rebnconvin = REBNCONV(in_ch,out_ch,dirate=1)

        self.rebnconv1 = REBNCONV(out_ch,mid_ch,dirate=1)
        self.pool1 = nn.AvgPool2d(2,stride=2,ceil_mode=True)

        self.rebnconv2 = REBNCONV(mid_ch,mid_ch,dirate=1)
        self.pool2 = nn.AvgPool2d(2,stride=2,ceil_mode=True)

        self.rebnconv3 = REBNCONV(mid_ch,mid_ch,dirate=1)

        self.rebnconv4 = REBNCONV(mid_ch,mid_ch,dirate=2)

        self.rebnconv3d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
        self.rebnconv2d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
        self.rebnconv1d = REBNCONV(mid_ch*2,out_ch,dirate=1)

    def forward(self,x,y):

        hx = x

        hxin = self.rebnconvin(hx)

        hx1 = self.rebnconv1(hxin)
        hx = self.pool1(hx1)

        hx2 = self.rebnconv2(hx)
        hx = self.pool2(hx2)

        hx3 = self.rebnconv3(hx)

        hx4 = self.rebnconv4(hx3)

        hx3d = self.rebnconv3d(torch.cat((hx4,hx3),1))
        hx3dup = _upsample_like(hx3d,hx2)

        hx2d = self.rebnconv2d(torch.cat((hx3dup,hx2),1))
        hx2dup = _upsample_like(hx2d,hx1)

        hx1d = self.rebnconv1d(torch.cat((hx2dup,hx1),1))

        return hx1d*y + hxin*(1-y)
### My_RSU-4F ###
class My_RSU4F(nn.Module):#UNet04FRES(nn.Module):

    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(My_RSU4F,self).__init__()

        self.rebnconvin = REBNCONV(in_ch,out_ch,dirate=1)

        self.rebnconv1 = REBNCONV(out_ch,mid_ch,dirate=1)
        self.rebnconv2 = REBNCONV(mid_ch,mid_ch,dirate=2)
        self.rebnconv3 = REBNCONV(mid_ch,mid_ch,dirate=4)

        self.rebnconv4 = REBNCONV(mid_ch,mid_ch,dirate=8)

        self.rebnconv3d = REBNCONV(mid_ch*2,mid_ch,dirate=4)
        self.rebnconv2d = REBNCONV(mid_ch*2,mid_ch,dirate=2)
        self.rebnconv1d = REBNCONV(mid_ch*2,out_ch,dirate=1)

    def forward(self,x,y):

        hx = x

        hxin = self.rebnconvin(hx)

        hx1 = self.rebnconv1(hxin)
        hx2 = self.rebnconv2(hx1)
        hx3 = self.rebnconv3(hx2)

        hx4 = self.rebnconv4(hx3)

        hx3d = self.rebnconv3d(torch.cat((hx4,hx3),1))
        hx2d = self.rebnconv2d(torch.cat((hx3d,hx2),1))
        hx1d = self.rebnconv1d(torch.cat((hx2d,hx1),1))

        return hx1d*y + hxin*(1-y)

class Fuse_area(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Fuse_area, self).__init__()
        self.Conv = nn.Sequential(
            depthwise_separable_conv(in_channels, out_channels, stride=1),
            nn.BatchNorm2d(out_channels),
            nn.Conv2d(out_channels, out_channels, 1),
        )
        self.Sigmoid = nn.Softmax(dim=2)
        self.conv_out = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, Detail_x, Semantic_x):
        DetailBranch  = self.Conv(Detail_x)
        DetailBranch_sig = self.Sigmoid(DetailBranch)
        SemanticBranch = self.Conv(Semantic_x)
        SemanticBranch_sig = self.Sigmoid(SemanticBranch)
        out1 = torch.matmul(DetailBranch_sig, SemanticBranch )
        out2 = torch.matmul(SemanticBranch_sig, DetailBranch)
        out = self.conv_out(out1 + out2)
        return out
class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )
    def forward(self, x):
        b, c, h, w = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return y.expand(b, c, 2*h, 2*w)
class FCtL3(nn.Module):
    def __init__(self, inplanes, planes):
        conv_nd = nn.Conv2d
        super(FCtL3, self).__init__()
        self.conv_value = conv_nd(inplanes, inplanes, kernel_size=1, bias=False)
        self.conv_value_1 = conv_nd(inplanes, inplanes, kernel_size=1, bias=False)
        self.conv_value_2 = conv_nd(inplanes, inplanes, kernel_size=1, bias=False)
        self.conv_out = None

        self.conv_query = conv_nd(inplanes, planes, kernel_size=1)
        self.conv_key = conv_nd(inplanes, planes, kernel_size=1)
        self.conv_query_1 = conv_nd(inplanes, planes, kernel_size=1)
        self.conv_key_1 = conv_nd(inplanes, planes, kernel_size=1)
        self.conv_query_2 = conv_nd(inplanes, planes, kernel_size=1)
        self.conv_key_2 = conv_nd(inplanes, planes, kernel_size=1)

        self.in_1 = conv_nd(inplanes, planes, kernel_size=1)
        self.in_2 = conv_nd(inplanes, planes, kernel_size=1)
        self.in_3 = conv_nd(inplanes, planes, kernel_size=1)
        self.trans = conv_nd(inplanes * 3, planes * 3, kernel_size=1)
        self.out_1 = conv_nd(inplanes, planes, kernel_size=1)
        self.out_2 = conv_nd(inplanes, planes, kernel_size=1)
        self.out_3 = conv_nd(inplanes, planes, kernel_size=1)

        self.softmax = nn.Softmax(dim=2)
        self.softmax_H = nn.Softmax(dim=0)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.gamma_1 = nn.Parameter(torch.zeros(1))
        self.gamma_2 = nn.Parameter(torch.zeros(1))

        self.weight_init_scale = 1.0

        self.reset_parameters()
        self.reset_weight_and_weight_decay()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d) or isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
                init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    init.zeros_(m.bias)
                m.inited = True

    def reset_weight_and_weight_decay(self):
        init.normal_(self.conv_query.weight, 0, 0.01 * self.weight_init_scale)
        init.normal_(self.conv_key.weight, 0, 0.01 * self.weight_init_scale)
        self.conv_query.weight.wd = 0.0
        self.conv_query.bias.wd = 0.0
        self.conv_key.weight.wd = 0.0
        self.conv_key.bias.wd = 0.0
    def forward(self, x, y=None, z=None):
        residual = x

        value = self.conv_value(y)
        value = value.view(value.size(0), value.size(1), -1)
        out_sim = None
        if z is not None:
            value_1 = self.conv_value_1(z)
            value_1 = value_1.view(value_1.size(0), value_1.size(1), -1)
            out_sim_1 = None
            value_2 = self.conv_value_2(x)
            value_2 = value_2.view(value_2.size(0), value_2.size(1), -1)
            out_sim_2 = None

        query = self.conv_query(x)
        key = self.conv_key(y)
        query = query.view(query.size(0), query.size(1), -1)
        key = key.view(key.size(0), key.size(1), -1)
        if z is not None:
            query_1 = self.conv_query_1(x)
            key_1 = self.conv_key_1(z)
            query_1 = query_1.view(query_1.size(0), query_1.size(1), -1)
            key_1 = key_1.view(key_1.size(0), key_1.size(1), -1)
            query_2 = self.conv_query_2(x)
            key_2 = self.conv_key_2(x)
            query_2 = query_2.view(query_2.size(0), query_2.size(1), -1)
            key_2 = key_2.view(key_2.size(0), key_2.size(1), -1)

        sim_map = torch.bmm(query.transpose(1, 2), key)
        sim_map = self.softmax(sim_map)
        out_sim = torch.bmm(sim_map, value.transpose(1, 2))
        out_sim = out_sim.transpose(1, 2)
        out_sim = out_sim.view(out_sim.size(0), out_sim.size(1), *x.size()[2:])
        out_sim = self.gamma * out_sim
        if z is not None:
            sim_map_1 = torch.bmm(query_1.transpose(1, 2), key_1)
            sim_map_1 = self.softmax(sim_map_1)
            out_sim_1 = torch.bmm(sim_map_1, value_1.transpose(1, 2))
            out_sim_1 = out_sim_1.transpose(1, 2)
            out_sim_1 = out_sim_1.view(out_sim_1.size(0), out_sim_1.size(1), *x.size()[2:])
            out_sim_1 = self.gamma_1 * out_sim_1
            sim_map_2 = torch.bmm(query_2.transpose(1, 2), key_2)
            sim_map_2 = self.softmax(sim_map_2)
            out_sim_2 = torch.bmm(sim_map_2, value_2.transpose(1, 2))
            out_sim_2 = out_sim_2.transpose(1, 2)
            out_sim_2 = out_sim_2.view(out_sim_2.size(0), out_sim_2.size(1), *x.size()[2:])
            out_sim_2 = self.gamma_2 * out_sim_2

        if z is not None:
            H_1 = self.in_1(out_sim)
            H_2 = self.in_2(out_sim_1)
            H_3 = self.in_3(out_sim_2)
            H_cat = torch.cat((H_1, H_2, H_3), 1)
            H_tra = self.trans(H_cat)
            H_spl = torch.split(H_tra, 64, dim=1)
            H_4 = torch.sigmoid(self.out_1(H_spl[0]))
            H_5 = torch.sigmoid(self.out_2(H_spl[1]))
            H_6 = torch.sigmoid(self.out_3(H_spl[2]))
            H_st = torch.stack((H_4, H_5, H_6), 0)
            H_all = self.softmax_H(H_st)
        if z is not None:
            out = residual + H_all[0] * out_sim + H_all[1] * out_sim_1 + H_all[2] * out_sim_2
        else:
            out = residual + out_sim
        return out
#相同尺寸，线性
class Fuse_line(nn.Module):
    def __init__(self, inplanes, planes):
        conv_nd = nn.Conv2d
        super(Fuse_line, self).__init__()
        self.conv_value = conv_nd(inplanes, inplanes, kernel_size=1, bias=False)
        self.conv_value_1 = conv_nd(inplanes, inplanes, kernel_size=1, bias=False)
        self.conv_value_2 = conv_nd(inplanes, inplanes, kernel_size=1, bias=False)
        self.conv_out = None

        self.conv_query = conv_nd(inplanes, planes, kernel_size=1)
        self.conv_key = conv_nd(inplanes, planes, kernel_size=1)
        self.conv_query_1 = conv_nd(inplanes, planes, kernel_size=1)
        self.conv_key_1 = conv_nd(inplanes, planes, kernel_size=1)
        self.conv_query_2 = conv_nd(inplanes, planes, kernel_size=1)
        self.conv_key_2 = conv_nd(inplanes, planes, kernel_size=1)

        self.in_1 = conv_nd(inplanes, planes, kernel_size=1)
        self.in_2 = conv_nd(inplanes, planes, kernel_size=1)
        self.in_3 = conv_nd(inplanes, planes, kernel_size=1)
        self.trans = conv_nd(inplanes * 2, planes * 2, kernel_size=1)
        self.out_1 = conv_nd(inplanes, planes, kernel_size=1)
        self.out_2 = conv_nd(inplanes, planes, kernel_size=1)
        self.out_3 = conv_nd(inplanes, planes, kernel_size=1)

        self.softmax = nn.Softmax(dim=2)
        self.softmax_H = nn.Softmax(dim=0)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.gamma_1 = nn.Parameter(torch.zeros(1))
        self.gamma_2 = nn.Parameter(torch.zeros(1))

        self.weight_init_scale = 1.0
        self.channel = inplanes

        self.reset_parameters()
        self.reset_weight_and_weight_decay()
        self.flatten = nn.Flatten(start_dim=2)
        self.fc_m = nn.Linear(9,36)
    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d) or isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
                init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    init.zeros_(m.bias)
                m.inited = True
    def reset_weight_and_weight_decay(self):
        init.normal_(self.conv_query.weight, 0, 0.01 * self.weight_init_scale)
        init.normal_(self.conv_key.weight, 0, 0.01 * self.weight_init_scale)
        self.conv_query.weight.wd = 0.0
        self.conv_query.bias.wd = 0.0
        self.conv_key.weight.wd = 0.0
        self.conv_key.bias.wd = 0.0
    def forward(self, x, y, z, m): #x,y,z代表语义分割结果
        y,z,m = map(self.conv_value, [y,z,m])
        y_line,z_line,m_line = map(self.flatten, [y,z,m]) # 144, 36, 9
        m_line = self.fc_m(m_line) # b,c,36
        m_line = m_line.permute(0,2,1)
        fuse_mz = torch.bmm(z_line,m_line)
        fuse_mzy = torch.bmm(fuse_mz,y_line) + y

        residual = x
        value = fuse_mzy
        value = value.view(value.size(0), value.size(1), -1)

        value_2 = self.conv_value_2(x)
        value_2 = value_2.view(value_2.size(0), value_2.size(1), -1)

        query = self.conv_query(x)
        key = self.conv_key(fuse_mzy)
        query = query.view(query.size(0), query.size(1), -1)
        key = key.view(key.size(0), key.size(1), -1)

        query_2 = self.conv_query_2(x)
        key_2 = self.conv_key_2(x)
        query_2 = query_2.view(query_2.size(0), query_2.size(1), -1)
        key_2 = key_2.view(key_2.size(0), key_2.size(1), -1)

        sim_map = torch.bmm(query.transpose(1, 2), key)
        sim_map = self.softmax(sim_map)
        out_sim = torch.bmm(sim_map, value.transpose(1, 2))
        out_sim = out_sim.transpose(1, 2)
        out_sim = out_sim.view(out_sim.size(0), out_sim.size(1), *x.size()[2:])
        out_sim = self.gamma * out_sim

        sim_map_2 = torch.bmm(query_2.transpose(1, 2), key_2)
        sim_map_2 = self.softmax(sim_map_2)
        out_sim_2 = torch.bmm(sim_map_2, value_2.transpose(1, 2))
        out_sim_2 = out_sim_2.transpose(1, 2)
        out_sim_2 = out_sim_2.view(out_sim_2.size(0), out_sim_2.size(1), *x.size()[2:])
        out_sim_2 = self.gamma_2 * out_sim_2

        H_1 = self.in_1(out_sim)
        H_3 = self.in_3(out_sim_2)
        H_cat = torch.cat((H_1, H_3), 1)
        H_tra = self.trans(H_cat)
        H_spl = torch.split(H_tra, self.channel, dim=1)
        H_5 = torch.sigmoid(self.out_2(H_spl[0]))
        H_6 = torch.sigmoid(self.out_3(H_spl[1]))
        H_st = torch.stack((H_5, H_6), 0)
        H_all = self.softmax_H(H_st)
        out = H_all[0] * out_sim + H_all[1] * out_sim_2 +residual
        return out

class AFNPBlock(nn.Module):
    def __init__(self, pool_size):
        super(AFNPBlock, self).__init__()
        self.pool_1 = nn.AdaptiveAvgPool2d(pool_size[0])
        self.pool_4 = nn.AdaptiveAvgPool2d(pool_size[1])
        self.pool_9 = nn.AdaptiveAvgPool2d(pool_size[2])
        self.Conv_Value = nn.Conv2d(512, 512, 1)
        self.Conv_Key = nn.Sequential(nn.Conv2d(512, 512, kernel_size=1, stride=1, padding=0),
                                      nn.BatchNorm2d(512),
                                      nn.ReLU()
                                      )
        self.Conv_Query = nn.Sequential(nn.Conv2d(512, 512, 1),
                                        nn.BatchNorm2d(512),
                                        nn.ReLU()
                                        )
        self.fc = nn.Sequential(nn.Linear(144, 72), nn.Dropout(0.2), nn.Linear(72, 144))
        self.ConvOut = nn.Conv2d(512,512,1)
        self.flatten = nn.Flatten(start_dim=2)
        # 给ConvOut初始化为0
        nn.init.constant_(self.ConvOut.weight, 0)
        nn.init.constant_(self.ConvOut.bias, 0)

    def forward(self, gx6, binary_16, binary_8, binary_4): # color, grey

        Query = self.Conv_Query(gx6)
        Query = Query.flatten(2).permute(0,2,1)
        key_16_9 = self.pool_9(binary_16)
        key_8_4 = self.pool_4(binary_8)
        key_4_1 = self.pool_1(binary_4)
        key_16_1 = self.pool_1(binary_16)

        Key = torch.cat([i for i in map(self.flatten, [key_16_9,binary_8,key_8_4,binary_4,key_4_1,key_16_1])],dim=-1)
        Value =  Key.contiguous().permute(0,2,1)
        Concat_QK = torch.matmul(Query, Key)
        Concat_QK = (512 ** -.5) * Concat_QK
        Concat_QK = F.softmax(Concat_QK, dim=-1)

        Aggregate_QKV = torch.matmul(Concat_QK, Value)
        # Aggregate_QKV = [batch, value_channels, h*w]
        Aggregate_QKV = Aggregate_QKV.permute(0, 2, 1).contiguous()
        # Aggregate_QKV = [batch, value_channels, h*w] -> [batch, value_channels, h, w]
        Aggregate_QKV = Aggregate_QKV.view(*gx6.shape)
        # Conv out
        Aggregate_QKV = self.ConvOut(Aggregate_QKV)

        return Aggregate_QKV+gx6
class channel_atten(nn.Module):
    def __init__(self, in_channels):
        super(channel_atten, self).__init__()
        self.module = nn.Sequential( nn.AdaptiveAvgPool2d(1), nn.Conv2d(in_channels, in_channels, 1), nn.BatchNorm2d(in_channels), nn.Sigmoid())
    def forward(self,x):
        res = x
        atten = self.module(x)
        out = torch.mul(atten,x) + res
        return out

### U^2-Net small ###
class U2NETP(nn.Module):

    def __init__(self,in_ch=3,out_ch=1):
        super(U2NETP,self).__init__()

        self.stage1 = RSU7(in_ch,32,64)
        self.pool12 = nn.MaxPool2d(2,stride=2,ceil_mode=True)
        self.stage2 = RSU6(64,32,128)
        self.pool23 = nn.MaxPool2d(2,stride=2,ceil_mode=True)
        self.stage3 = RSU5(128,64,256)
        self.pool34 = nn.AvgPool2d(2,stride=2,ceil_mode=True)
        self.stage4 = RSU4(256,128,512)
        self.pool45 = nn.AvgPool2d(2,stride=2,ceil_mode=True)
        self.stage5 = RSU4F(512,256,512)
        self.pool56 = nn.AvgPool2d(2,stride=2,ceil_mode=True)
        self.stage6 = RSU4F(512,256,512)

        self.stage11 = RSU7(in_ch,32,64)
        self.stage22 = RSU6(64,32,128)
        self.stage33 = RSU5(128,64,256)
        self.stage44 = RSU4(256,128,512)
        self.stage55 = RSU4F(512,256,512)
        self.stage66 = RSU4F(512,256,512)

        self.stage1_my = My_RSU7(in_ch,32,64)
        self.stage2_my = My_RSU6(64,32,128)
        self.stage3_my = My_RSU_5(128,64,256)
        self.stage4_my = My_RSU_4(256,128,512)
        self.stage5_my = My_RSU4F(512,256,512)
        self.stage6_my = My_RSU4F(512,256,512)

        # decoder
        self.stage5d = RSU4F(1024,256,512)
        self.stage4d = RSU4(1024,128,256)
        self.stage3d = RSU5(512,64,128)
        self.stage2d = RSU6(256,32,64)
        self.stage1d = RSU7(128,16,64)
        self.stage55d = RSU4F(1024,256,512)
        self.stage44d = RSU4(1024,128,256)
        self.stage33d = RSU5(512,64,128)
        self.stage22d = RSU6(256,32,64)
        self.stage11d = RSU7(128,16,64)


        self.side1 = nn.Conv2d(64,out_ch,3,padding=1)
        self.side2 = nn.Conv2d(128,out_ch,3,padding=1)
        self.side3 = nn.Conv2d(256,out_ch,3,padding=1)
        self.side4 = nn.Conv2d(512,out_ch,3,padding=1)
        self.side5 = nn.Conv2d(512,out_ch,3,padding=1)
        self.side6 = nn.Conv2d(512,out_ch,3,padding=1)

        self.side11 = nn.Conv2d(64,out_ch,3,padding=1)
        self.side22 = nn.Conv2d(64,out_ch,3,padding=1)
        self.side33 = nn.Conv2d(128,out_ch,3,padding=1)
        self.side44 = nn.Conv2d(256,out_ch,3,padding=1)
        self.side55 = nn.Conv2d(512,out_ch,3,padding=1)
        self.side66 = nn.Conv2d(512,out_ch,3,padding=1)


        self.outconv = nn.Conv2d(6*out_ch,out_ch,1)


        self.up_conv = UpConv(64,64)
        self.up_conv_4 = UpConv(1, 1,4)
        self.conv_out = nn.Sequential(
            nn.Conv2d(64*2, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),  nn.ReLU(inplace=True)
        )
        self.sig = nn.Sigmoid()
        self.soft = nn.Softmax(dim=2)
        self.fuse_line = AFNPBlock([1,4,9])
        self.channel_atten1 = channel_atten(64)
        self.channel_atten2 = channel_atten(128)
        self.channel_atten3 = channel_atten(512)

    def forward(self, grey, color):
        ############################################  多光谱图像编码  #####################################################
        cx = color
        cx1 = self.stage1(cx)  # 128*128   96
        cx = self.pool12(cx1)
        cx2 = self.stage2(cx)  # 64*64     48
        cx = self.pool23(cx2)
        cx3 = self.stage3(cx)  # 32*32     24
        cx = self.pool34(cx3)
        cx4 = self.stage4(cx)  # 16*16     12
        cx = self.pool45(cx4) 
        cx5 = self.stage5(cx)  # 8*8        6
        cx = self.pool56(cx5)
        cx6 = self.stage6(cx)  # 4*4        3


        side_512_cx, side_256_cx, side_128_cx = self.side1(cx1), self.side2(cx2), self.side3(cx3)
        binary_128, binary_64, binary_32, binary_16, binary_8, binary_4 = map(self.soft,
                                                                              [side_512_cx, side_256_cx, side_128_cx, cx4, cx5, cx6])
        binary_128_512 = F.interpolate(binary_128, scale_factor=4, mode='bilinear', align_corners=True)
        ############################################  全色图像编码  #####################################################
        gx = grey
        gx1 = self.stage11(gx)  # 512*512
        gx = self.pool12(gx1)
        gx2 = self.stage22(gx)  # 256*256
        gx = self.pool23(gx2)
        gx3 = self.stage3_my(gx, binary_128)  # 128*128
        gx = self.pool34(gx3)
        gx4 = self.stage4_my(gx, binary_64)  # 64*64
        gx = self.pool45(gx4)
        gx5 = self.stage5_my(gx, binary_32)  # 32*32
        gx = self.pool56(gx5)
        gx6 = self.fuse_line(gx, binary_16,binary_8,binary_4)  # 16*16
        gx6 = self.channel_atten3(gx6)
        gx6up = _upsample_like(gx6, gx5)
        ############################################  特征融合  #####################################################

        ############################################      解码     #####################################################
        gx5d = self.stage55d(torch.cat((gx6up, gx5), 1))
        gx5dup = _upsample_like(gx5d, gx4)
        gx4d = self.stage44d(torch.cat((gx5dup, gx4), 1))
        gx4dup = _upsample_like(gx4d, gx3)
        gx3d = self.stage33d(torch.cat((gx4dup, gx3), 1))
        gx3dup = _upsample_like(gx3d, gx2)
        gx2d = self.stage22d(torch.cat((gx3dup, gx2), 1))
        gx2dup = _upsample_like(gx2d, gx1)
        gx1d = self.stage11d(torch.cat((gx2dup, gx1), 1))
        # side output
        side_512, side_256, side_128, side_64, side_32, side_16 = self.side11(gx1d), self.side22(gx2d), self.side33(gx3d), \
                                                                  self.side44(gx4d), self.side55(gx5d), self.side66(gx6)
        side_256, side_128, side_64, side_32, side_16 = map(_upsample_like, *zip([side_256, side_512],
                                                                                 [side_128, side_512],
                                                                                 [side_64, side_512],
                                                                                 [side_32, side_512],
                                                                                 [side_16, side_512]))

        d0 = self.outconv(torch.cat((side_512, side_256, side_128, side_64, side_32, side_16), 1))

        m0, m1, m2, m3, m4, m5, m6 = map(self.sig, [d0, side_512, side_256, side_128, side_64, side_32, side_16])

        return m0, m1, m2, m3, m4, m5, m6, binary_128_512

    """
    高分辨率为指导分割，然后使用多光谱图像做细节分割优化，其中stage阶段被改写
            # a = self.side6(fusion_64)
        # b = self.side6(fusion_64)
        # a1 = self.side6(cx2)
        # b1 = self.side6(gx4)
        # a = numpy.array(a[0][0].cpu().detach())
        # b = numpy.array(b[0][0].cpu().detach())
        # a1 = numpy.array(a1[0][0].cpu().detach())
        # b1 = numpy.array(b1[0][0].cpu().detach())
        #
        # plt.figure()
        # plt.subplot(2, 2, 1)
        # plt.imshow(a)
        # plt.subplot(2, 2, 2)
        # plt.imshow(b)
        # plt.subplot(2, 2, 3)
        # plt.imshow(a1)
        # plt.subplot(2, 2, 4)
        # plt.imshow(b1)
        # plt.show()
    """