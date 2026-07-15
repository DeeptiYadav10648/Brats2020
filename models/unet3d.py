import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import GCNConv


class GCNRefiner(nn.Module):

    def __init__(
        self,
        in_channels=256,
        hidden_channels=256
    ):

        super().__init__()

        self.conv1 = GCNConv(
            in_channels,
            hidden_channels
        )

        self.bn1 = nn.BatchNorm1d(
            hidden_channels
        )

        self.conv2 = GCNConv(
            hidden_channels,
            hidden_channels
        )

        self.bn2 = nn.BatchNorm1d(
            hidden_channels
        )

        self.dropout = 0.3

    def forward(
        self,
        x,
        edge_index
    ):

        residual = x

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

        x = self.conv2(
            x,
            edge_index
        )

        x = self.bn2(x)

        x = F.relu(x)

        x = x + residual

        return x