import torch

from torch_geometric.data import Data


def create_graph(
    nodes,
    edge_index,
    labels
):

    x = torch.tensor(
        nodes,
        dtype=torch.float
    )

    edge_index = torch.tensor(
        edge_index,
        dtype=torch.long
    )

    y = torch.tensor(
        labels,
        dtype=torch.long
    )

    graph = Data(
        x=x,
        edge_index=edge_index,
        y=y
    )

    return graph