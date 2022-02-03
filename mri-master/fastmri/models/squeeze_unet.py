"""
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import torch
from torch import nn
from torch.nn import functional as F

"""
From Pytorch SqueezeNet Fire Module Implementation:
https://github.com/pytorch/vision/blob/master/torchvision/models/squeezenet.py
"""
class Fire(nn.Module):
    def __init__(
        self,
        inplanes: int,
        squeeze_planes: int,
        expand1x1_planes: int,
        expand3x3_planes: int
    ) -> None:
        super(Fire, self).__init__()
        self.inplanes = inplanes
        self.squeeze = nn.Conv2d(inplanes, squeeze_planes, kernel_size=1)
        self.squeeze_activation = nn.ReLU(inplace=True)
        self.expand1x1 = nn.Conv2d(squeeze_planes, expand1x1_planes, kernel_size=1)
        self.expand1x1_activation = nn.ReLU(inplace=True)
        self.expand3x3 = nn.Conv2d(squeeze_planes, expand3x3_planes, kernel_size=3, padding=1)
        self.expand3x3_activation = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.squeeze_activation(self.squeeze(x))
        return torch.cat([self.expand1x1_activation(self.expand1x1(x)), self.expand3x3_activation(self.expand3x3(x))], 1)


class SqueezeUnet(nn.Module):
    """
    PyTorch implementation of a Squeeze U-Net model.

    N. Beheshti and L. Johnsson, "Squeeze U-Net: A Memory and Energy Efficient 
    Image Segmentation Network," 2020 IEEE/CVF Conference on Computer Vision and 
    Pattern Recognition Workshops (CVPRW), 2020, pp. 1495-1504, 
    doi: 10.1109/CVPRW50498.2020.00190.

    Adapted from:
    Leonardo lontra
    https://github.com/lhelontra/squeeze-unet/blob/master/squeezeunet.py

    Supported by:
    https://github.com/pytorch/vision/blob/master/torchvision/models/squeezenet.py
    """

    def __init__(
        self,
        in_chans: int,
        out_chans: int,
        chans: int = 32,
        deconv_ksize=3,
        drop_prob: float = 0.5,
    ):
        """
        Args:
            in_chans: Number of channels in the input to the U-Net model.
            out_chans: Number of channels in the output to the U-Net model.
            chans: Number of output channels of the first convolution layer.
            num_pool_layers: Number of down-sampling and up-sampling layers.
            drop_prob: Dropout probability.
        """
        super().__init__()

        self.drop_prob = drop_prob
        self.deconv_ksize = deconv_ksize

        self.x01 = nn.Conv2d(in_chans, 64, kernel_size=3, stride=2, padding=1)
        self.relu = nn.ReLU(inplace=False)
        pool_size = 3
        pool_padding = int((pool_size-1)/2)
        self.x02 = nn.MaxPool2d(kernel_size=(pool_size, pool_size), stride=2, padding=pool_padding, ceil_mode=True)
        self.x03 = Fire(inplanes=64, squeeze_planes=16, expand1x1_planes=64, expand3x3_planes=64)
        self.x04 = Fire(inplanes=64, squeeze_planes=16, expand1x1_planes=64, expand3x3_planes=64)
        self.x05 = nn.MaxPool2d(kernel_size=pool_size, stride=2, padding=pool_padding, ceil_mode=True)
        self.x06 = Fire(inplanes=128, squeeze_planes=32, expand1x1_planes=128, expand3x3_planes=128)
        self.x07 = Fire(inplanes=256, squeeze_planes=32, expand1x1_planes=128, expand3x3_planes=128)
        self.x08 = nn.MaxPool2d(kernel_size=pool_size, stride=2, padding=pool_padding, ceil_mode=True)
        self.x09 = Fire(inplanes=256, squeeze_planes=48, expand1x1_planes=192, expand3x3_planes=192)
        self.x10 = Fire(inplanes=384, squeeze_planes=48, expand1x1_planes=192, expand3x3_planes=192)
        self.x11 = Fire(inplanes=384, squeeze_planes=64, expand1x1_planes=256, expand3x3_planes=256)
        self.x12 = Fire(inplanes=512, squeeze_planes=64, expand1x1_planes=256, expand3x3_planes=256)

        transpose_padding = int((self.deconv_ksize-1)/2)
        self.t1 = nn.ConvTranspose2d(in_channels=512, out_channels=192, kernel_size=self.deconv_ksize, stride=1, padding=transpose_padding)
        self.t2 = nn.ConvTranspose2d(in_channels=384, out_channels=128, kernel_size=self.deconv_ksize, stride=1, padding=transpose_padding)
        self.t3 = nn.ConvTranspose2d(in_channels=256, out_channels=64, kernel_size=self.deconv_ksize, stride=2, padding=transpose_padding)
        self.t4 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=self.deconv_ksize, stride=2, padding=transpose_padding)

        self.f1 = Fire(inplanes=576, squeeze_planes=48, expand1x1_planes=192, expand3x3_planes=192)
        self.f2 = Fire(inplanes=384, squeeze_planes=32, expand1x1_planes=128, expand3x3_planes=128)
        self.f3 = Fire(inplanes=192, squeeze_planes=16, expand1x1_planes=64, expand3x3_planes=64)
        self.f4 = Fire(inplanes=128, squeeze_planes=16, expand1x1_planes=32, expand3x3_planes=32)

        self.x1 = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1)
        self.x3 = nn.Conv2d(64, out_chans, kernel_size=1)
        self.x4 = nn.Sigmoid()

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image: Input 4D tensor of shape `(N, in_chans, H, W)`.

        Returns:
            Output tensor of shape `(N, out_chans, H, W)`.
        """
        #variables are named o[stepnumber]
        #to help with following the steps
        o1 = self.x01(image)
        o2 = self.relu(o1)
        o3 = self.x02(o2)
        o4 = self.x04(o3)
        o5 = self.x05(o4)
        o6 = self.x06(o5)
        o7 = self.x07(o6)
        o8 = self.x08(o7)
        o9 = self.x09(o8)
        o10 = self.x10(o9)
        o11 = self.x11(o10)
        o12 = self.x12(o11)
        o13 = torch.dropout(o12, p=self.drop_prob, train=True)

        o14 = self.t1(o13)
        o15 = torch.cat([o14, o10], dim=1)
        o16 = self.f1(o15)

        o17 = self.t2(o16)
        o18 = torch.cat([o17, o8], dim=1)
        o19 = self.f2(o18)

        o20 = self.t3(o19)
        o21 = torch.cat([o20, o5], dim=1)
        o22 = self.f3(o21)

        o23 = self.t4(o22)
        o24 = torch.cat([o23, o3], dim=1)
        o25 = self.f4(o24)

        upsample_4 = nn.Upsample(size=(o1.shape[2]))
        o26 = upsample_4(o25)
        o27 = torch.cat([o26, o1], dim=1)
        o28 = self.x1(o27)
        o29 = self.relu(o28)
        x2 = nn.Upsample(size=(image.shape[2], image.shape[3]))
        o30 = x2(o29)
        o31 = self.x3(o30)
        output = self.x4(o31)  # Final activation is sigmoid

        return output
