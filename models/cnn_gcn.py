import torch
import torch.nn.functional as F

from torch_geometric.nn import GCNConv


class HybridGCN(torch.nn.Module):

    def __init__(
        self,
        in_channels=64,
        hidden=128,
        num_classes=4
    ):

        super().__init__()

        self.conv1 = GCNConv(
            in_channels,
            hidden
        )

        self.bn1 = torch.nn.BatchNorm1d(hidden)

        self.conv2 = GCNConv(
            hidden,
            hidden
        )

        self.bn2 = torch.nn.BatchNorm1d(hidden)

        self.conv3 = GCNConv(
            hidden,
            num_classes
        )

        self.dropout = 0.3

    def forward(
        self,
        x,
        edge_index
    ):

        x = self.conv1(
            x,
            edge_index
        )

        x = self.bn1(x)

        x = F.relu(x)

        x = F.dropout(
            x,
            p=self.dropout,
            training=self.training
        )

        residual = x

        x = self.conv2(
            x,
            edge_index
        )

        x = self.bn2(x)

        x = F.relu(x)

        x = F.dropout(
            x,
            p=self.dropout,
            training=self.training
        )

        x = x + residual

        x = self.conv3(
            x,
            edge_index
        )

        return x