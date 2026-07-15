import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock3D(nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels
    ):

        super().__init__()

        self.conv1 = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            bias=False
        )

        self.bn1 = nn.BatchNorm3d(
            out_channels
        )

        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv3d(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            bias=False
        )

        self.bn2 = nn.BatchNorm3d(
            out_channels
        )

        if in_channels != out_channels:

            self.shortcut = nn.Sequential(

                nn.Conv3d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    bias=False
                ),

                nn.BatchNorm3d(
                    out_channels
                )

            )

        else:

            self.shortcut = nn.Identity()

    def forward(
        self,
        x
    ):

        identity = self.shortcut(x)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)

        x += identity

        x = self.relu(x)

        return x


class UpBlock(nn.Module):

    def __init__(
        self,
        in_channels,
        skip_channels,
        out_channels
    ):

        super().__init__()

        self.up = nn.ConvTranspose3d(

            in_channels,

            out_channels,

            kernel_size=2,

            stride=2

        )

        self.conv = ResidualBlock3D(

            out_channels + skip_channels,

            out_channels

        )

    def forward(

        self,

        x,

        skip

    ):

        x = self.up(x)

        if x.shape[2:] != skip.shape[2:]:

            x = F.interpolate(

                x,

                size=skip.shape[2:],

                mode="trilinear",

                align_corners=False

            )

        x = torch.cat(

            [x, skip],

            dim=1

        )

        x = self.conv(x)

        return x


class Decoder3D(nn.Module):

    def __init__(

        self,

        num_classes=4

    ):

        super().__init__()

        self.up4 = UpBlock(

            512,

            256,

            256

        )

        self.up3 = UpBlock(

            256,

            128,

            128

        )

        self.up2 = UpBlock(

            128,

            64,

            64

        )

        self.up1 = UpBlock(

            64,

            32,

            32

        )

        self.out_conv = nn.Conv3d(

            32,

            num_classes,

            kernel_size=1

        )

    def forward(

        self,

        bottleneck,

        x4,

        x3,

        x2,

        x1

    ):

        x = self.up4(

            bottleneck,

            x4

        )

        x = self.up3(

            x,

            x3

        )

        x = self.up2(

            x,

            x2

        )

        x = self.up1(

            x,

            x1

        )

        x = self.out_conv(

            x

        )

        return x


if __name__ == "__main__":

    device = torch.device(

        "cuda"

        if torch.cuda.is_available()

        else "cpu"

    )

    model = Decoder3D().to(device)

    bottleneck = torch.randn(

        1,

        512,

        8,

        8,

        8

    ).to(device)

    x4 = torch.randn(

        1,

        256,

        16,

        16,

        16

    ).to(device)

    x3 = torch.randn(

        1,

        128,

        32,

        32,

        32

    ).to(device)

    x2 = torch.randn(

        1,

        64,

        64,

        64,

        64

    ).to(device)

    x1 = torch.randn(

        1,

        32,

        128,

        128,

        128

    ).to(device)

    out = model(

        bottleneck,

        x4,

        x3,

        x2,

        x1

    )

    print(out.shape)