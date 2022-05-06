import torch
from torch import nn
from torchvision.models import resnet50, resnet18, resnet34
from torch import einsum
import torch.nn.functional as F
# from resnet import resnet34
try:
    from itertools import ifilterfalse
except ImportError:
    from itertools import filterfalse as ifilterfalse


class ConvRelu(nn.Module):
    def __init__(self, in_, out):
        super().__init__()
        self.conv = nn.Conv2d(in_, out, 3, padding=1)
        self.bn = nn.BatchNorm2d(out)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)

        return x


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DecoderBlock, self).__init__()
        self.in_channels = in_channels

        self.block = nn.Sequential(
            ConvRelu(in_channels, out_channels),
            ConvRelu(out_channels, out_channels),
            ConvRelu(out_channels, out_channels)
        )

    def forward(self, x):
        return self.block(x)


class UnetOverResnet18(nn.Module):
    def __init__(self, num_up_filters=512, pretrained=True):
        super().__init__()
        self.pool = nn.MaxPool2d(2, 2)
        self.up_sample = nn.functional.interpolate

        encoder = resnet18(pretrained=pretrained)

        self.conv1 = nn.Sequential(
            encoder.conv1,
            encoder.bn1,
            encoder.relu
        )
        self.conv2 = encoder.layer1
        self.conv3 = encoder.layer2
        self.conv4 = encoder.layer3
        self.conv5 = encoder.layer4

        self.center = DecoderBlock(512, num_up_filters)
        self.dec5 = DecoderBlock(512 + num_up_filters, num_up_filters // 2)
        self.dec4 = DecoderBlock(256 + num_up_filters // 2, num_up_filters // 4)
        self.dec3 = DecoderBlock(128 + num_up_filters // 4, num_up_filters // 8)
        self.dec2 = DecoderBlock(64 + num_up_filters // 8, num_up_filters // 16)
        self.dec1 = DecoderBlock(64 + num_up_filters // 16, num_up_filters // 32)
        self.dec0 = ConvRelu(num_up_filters // 32, num_up_filters // 32)

        self.final = nn.Conv2d(num_up_filters // 32, 3, kernel_size=1)

    def forward(self, x):
        conv1 = self.conv1(x)
        conv1_pool = self.pool(conv1)
        conv2 = self.conv2(conv1_pool)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv5 = self.conv5(conv4)

        center = self.center(self.pool(conv5))

        dec5 = self.dec5(torch.cat([self.up_sample(center, scale_factor=2), conv5], 1))
        dec4 = self.dec4(torch.cat([self.up_sample(dec5, scale_factor=2), conv4], 1))
        dec3 = self.dec3(torch.cat([self.up_sample(dec4, scale_factor=2), conv3], 1))
        dec2 = self.dec2(torch.cat([self.up_sample(dec3, scale_factor=2), conv2], 1))
        dec1 = self.dec1(torch.cat([self.up_sample(dec2, scale_factor=2), conv1], 1))

        dec0 = self.dec0(self.up_sample(dec1, scale_factor=2))

        x_out = self.final(dec0)

        return x_out


class Unet34(nn.Module):
    def __init__(self, num_up_filters=512, pretrained=True):
        super().__init__()
        self.pool = nn.MaxPool2d(2, 2)

        encoder = resnet34(pretrained=pretrained)

        self.relu = nn.ReLU(inplace=True)

        self.conv1 = nn.Sequential(encoder.conv1,
                                   encoder.bn1,
                                   self.relu)

        self.conv2 = encoder.layer1

        self.conv3 = encoder.layer2

        self.conv4 = encoder.layer3

        self.conv5 = encoder.layer4

        self.center = DecoderBlock(512, num_up_filters)
        self.dec5 = DecoderBlock(512 + num_up_filters, num_up_filters // 2)
        self.dec4 = DecoderBlock(256 + num_up_filters // 2, num_up_filters // 4)
        self.dec3 = DecoderBlock(128 + num_up_filters // 4, num_up_filters // 8)
        self.dec2 = DecoderBlock(64 + num_up_filters // 8, num_up_filters // 16)
        self.dec1 = DecoderBlock(64 + num_up_filters // 16, num_up_filters // 32)
        self.dec0 = ConvRelu(num_up_filters // 32, num_up_filters // 32)
        self.final = nn.Conv2d(num_up_filters // 32, 3, kernel_size=1)

        self.up_sample = nn.functional.interpolate

    def forward(self, x):
        conv1 = self.conv1(x)
        conv1_pool = self.pool(conv1)
        conv2 = self.conv2(conv1_pool)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv5 = self.conv5(conv4)

        center = self.center(self.pool(conv5))

        dec5 = self.dec5(torch.cat([self.up_sample(center, scale_factor=2), conv5], 1))
        dec4 = self.dec4(torch.cat([self.up_sample(dec5, scale_factor=2), conv4], 1))
        dec3 = self.dec3(torch.cat([self.up_sample(dec4, scale_factor=2), conv3], 1))
        dec2 = self.dec2(torch.cat([self.up_sample(dec3, scale_factor=2), conv2], 1))
        dec1 = self.dec1(torch.cat([self.up_sample(dec2, scale_factor=2), conv1], 1))

        dec0 = self.dec0(self.up_sample(dec1, scale_factor=2))

        x_out = self.final(dec0)

        return x_out


if __name__ == '__main__':
    model = UnetOverResnet18(pretrained=False)
    y = torch.zeros((1, 3, 256, 256))
    x = model(y)
    print(x.size())