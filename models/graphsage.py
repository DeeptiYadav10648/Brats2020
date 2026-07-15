import torch
import torch.nn.functional as F

from torch_geometric.nn import SAGEConv


class GraphSAGE(torch.nn.Module):

    def __init__(
        self,
        in_channels=16,
        hidden=128,
        num_classes=4,
        dropout=0.3
    ):

        super().__init__()

        self.dropout = dropout

        self.conv1 = SAGEConv(
            in_channels,
            hidden
        )

        self.conv2 = SAGEConv(
            hidden,
            hidden
        )

        self.conv3 = SAGEConv(
            hidden,
            num_classes
        )

    def forward(
        self,
        x,
        edge_index
    ):

        x1 = self.conv1(
            x,
            edge_index
        )

        x1 = F.relu(x1)

        x1 = F.dropout(
            x1,
            p=self.dropout,
            training=self.training
        )

        x2 = self.conv2(
            x1,
            edge_index
        )

        x2 = F.relu(x2)

        x2 = F.dropout(
            x2,
            p=self.dropout,
            training=self.training
        )

        # Residual connection
        x = x1 + x2

        x = self.conv3(
            x,
            edge_index
        )

        return x