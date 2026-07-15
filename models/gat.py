import torch
import torch.nn.functional as F

from torch_geometric.nn import GATv2Conv


class GAT(torch.nn.Module):

    def __init__(
        self,
        in_channels=16,
        hidden=64,
        num_classes=4,
        heads=4,
        dropout=0.3
    ):
        super().__init__()

        self.dropout = dropout

        self.conv1 = GATv2Conv(
            in_channels,
            hidden,
            heads=heads,
            dropout=dropout
        )

        self.conv2 = GATv2Conv(
            hidden * heads,
            hidden,
            heads=heads,
            dropout=dropout
        )

        self.conv3 = GATv2Conv(
            hidden * heads,
            num_classes,
            heads=1,
            concat=False,
            dropout=dropout
        )

    def forward(
        self,
        x,
        edge_index
    ):

        x = self.conv1(
            x,
            edge_index
        )

        x = F.elu(x)

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

        x = F.elu(x)

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