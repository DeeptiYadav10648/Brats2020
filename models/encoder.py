import torch
import torch.nn as nn


class ResidualBlock3D(nn.Module):
    """
    Residual Block:
    Conv3D -> BN -> ReLU
    Conv3D -> BN
    + Identity
    """

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

        self.relu = nn.ReLU(
            inplace=True
        )

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

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        out = self.relu(out)

        return out


class DownBlock(nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels
    ):

        super().__init__()

        self.block = ResidualBlock3D(
            in_channels,
            out_channels
        )

        self.pool = nn.MaxPool3d(
            kernel_size=2,
            stride=2
        )

    def forward(
        self,
        x
    ):

        feat = self.block(x)

        down = self.pool(feat)

        return feat, down


class Encoder3D(nn.Module):

    """
    Input:
        B x 4 x 128 x 128 x 128

    Outputs

    x1 : 32 x128³
    x2 : 64 x64³
    x3 :128 x32³
    x4 :256 x16³
    b  :512 x8³
    """

    def __init__(
        self,
        in_channels=4,
        base_channels=32
    ):

        super().__init__()

        self.down1 = DownBlock(
            in_channels,
            base_channels
        )

        self.down2 = DownBlock(
            base_channels,
            base_channels * 2
        )

        self.down3 = DownBlock(
            base_channels * 2,
            base_channels * 4
        )

        self.down4 = DownBlock(
            base_channels * 4,
            base_channels * 8
        )

        self.bottleneck = ResidualBlock3D(
            base_channels * 8,
            base_channels * 16
        )

    def forward(
        self,
        x
    ):

        x1, x = self.down1(x)

        x2, x = self.down2(x)

        x3, x = self.down3(x)

        x4, x = self.down4(x)

        b = self.bottleneck(x)

        return (
            x1,
            x2,
            x3,
            x4,
            b
        )


if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = Encoder3D().to(device)

    x = torch.randn(
        1,
        4,
        128,
        128,
        128
    ).to(device)

    x1, x2, x3, x4, b = model(x)

    print("x1 :", x1.shape)
    print("x2 :", x2.shape)
    print("x3 :", x3.shape)
    print("x4 :", x4.shape)
    print("b  :", b.shape)