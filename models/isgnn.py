import torch
import torch.nn.functional as F

from torch.nn import BatchNorm1d, Linear
from torch_geometric.nn import GCNConv, GATConv


class ResidualBlock(torch.nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels,
        dropout=0.3
    ):
        super().__init__()

        self.conv = GCNConv(
            in_channels,
            out_channels
        )

        self.bn = BatchNorm1d(
            out_channels
        )

        self.dropout = dropout

        if in_channels != out_channels:

            self.residual = Linear(
                in_channels,
                out_channels
            )

        else:

            self.residual = None

    def forward(
        self,
        x,
        edge_index
    ):

        identity = x

        out = self.conv(
            x,
            edge_index
        )

        out = self.bn(out)

        out = F.relu(out)

        out = F.dropout(
            out,
            p=self.dropout,
            training=self.training
        )

        if self.residual is not None:

            identity = self.residual(
                identity
            )

        out = out + identity

        return out


class ISGNN(torch.nn.Module):

    def __init__(
        self,
        in_channels=16,
        hidden=128,
        num_classes=4,
        dropout=0.3
    ):
        super().__init__()

        self.block1 = ResidualBlock(
            in_channels,
            hidden,
            dropout
        )

        self.block2 = ResidualBlock(
            hidden,
            hidden,
            dropout
        )

        self.attention = GATConv(
            hidden,
            hidden // 4,
            heads=4,
            concat=True,
            dropout=dropout
        )

        self.bn_attn = BatchNorm1d(
            hidden
        )

        self.block3 = ResidualBlock(
            hidden,
            hidden // 2,
            dropout
        )

        self.classifier = Linear(
            hidden // 2,
            num_classes
        )

    def forward(
        self,
        x,
        edge_index
    ):

        x = self.block1(
            x,
            edge_index
        )

        x = self.block2(
            x,
            edge_index
        )

        x = self.attention(
            x,
            edge_index
        )

        x = self.bn_attn(x)

        x = F.relu(x)

        x = self.block3(
            x,
            edge_index
        )

        x = self.classifier(x)

        return x