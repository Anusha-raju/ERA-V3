
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.convblock1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=24, kernel_size=(3, 3), padding=1, bias=False, dilation=2),
            nn.BatchNorm2d(24),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(in_channels=24, out_channels=48, kernel_size=(3, 3), padding=1, bias=False, dilation=2),
            nn.BatchNorm2d(48),
            nn.ReLU(),
            nn.Dropout(0.1),
            self.depthwise_separable_conv(48, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(0.1),
            self.depthwise_separable_conv(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )


        self.convblock2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=24, kernel_size=(3, 3), padding=1, bias=False, dilation=2),
            nn.BatchNorm2d(24),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(in_channels=24, out_channels=48, kernel_size=(3, 3), padding=1, bias=False, dilation=2),
            nn.BatchNorm2d(48),
            nn.ReLU(),
            nn.Dropout(0.1),
            self.depthwise_separable_conv(48, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(0.1),
            self.depthwise_separable_conv(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        
        self.convblock3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=24, kernel_size=(3, 3), padding=1, bias=False, dilation=2),
            nn.BatchNorm2d(24),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(in_channels=24, out_channels=48, kernel_size=(3, 3), padding=1, bias=False, dilation=2),
            nn.BatchNorm2d(48),
            nn.ReLU(),
            nn.Dropout(0.1),
            self.depthwise_separable_conv(48, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(0.1),
            self.depthwise_separable_conv(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        
        self.convblock4 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=24, kernel_size=(3, 3), padding=1, bias=False, dilation=2),
            nn.BatchNorm2d(24),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(in_channels=24, out_channels=48, kernel_size=(3, 3), padding=1, bias=False, dilation=2),
            nn.BatchNorm2d(48),
            nn.ReLU(),
            nn.Dropout(0.1),
            self.depthwise_separable_conv(48, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(0.1),
            self.depthwise_separable_conv(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        self.out = nn.Sequential(
            nn.AdaptiveMaxPool2d(1),
            nn.Conv2d(64, 10, kernel_size=1),
        )


    def depthwise_separable_conv(self, in_channels, out_channels, kernel_size=3, padding=1):
        """Depthwise Separable Convolution (Depthwise + Pointwise)"""
        # Depthwise Convolution
        depthwise_conv = nn.Conv2d(in_channels=in_channels, out_channels=in_channels,
                                   kernel_size=kernel_size, padding=padding, groups=in_channels,
                                   bias=False)

        # Pointwise Convolution (1x1)
        pointwise_conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                    kernel_size=1, padding=0, bias=False)

        return nn.Sequential(depthwise_conv, pointwise_conv)

    def forward(self, x):
      x = self.convblock1(x)
      x = self.convblock2(x)
      x = self.convblock3(x)
      x = self.convblock4(x)
      x = self.out(x)
      x = x.view(-1, 10)
      return F.log_softmax(x, dim=-1)