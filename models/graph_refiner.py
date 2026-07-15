import torch
import torch.nn as nn

from torch_geometric.data import Data
from torch_geometric.nn import GCNConv


class GraphRefiner(nn.Module):

    def __init__(
        self,
        channels=512
    ):

        super().__init__()

        self.gcn1 = GCNConv(
            channels,
            channels
        )

        self.gcn2 = GCNConv(
            channels,
            channels
        )

        self.relu = nn.ReLU(inplace=True)

    def build_graph(
        self,
        feature_map
    ):
        """
        feature_map:
        [B, C, D, H, W]
        """

        B, C, D, H, W = feature_map.shape

        graphs = []

        for b in range(B):

            x = feature_map[b]

            # -----------------------
            # Flatten voxels
            # -----------------------

            nodes = x.reshape(
                C,
                -1
            ).t()

            edge_src = []
            edge_dst = []

            def node_id(z, y, x):

                return (
                    z * H * W
                    + y * W
                    + x
                )

            # -----------------------
            # 6-neighbour graph
            # -----------------------

            for z in range(D):

                for y in range(H):

                    for x0 in range(W):

                        u = node_id(
                            z,
                            y,
                            x0
                        )

                        if x0 + 1 < W:

                            v = node_id(
                                z,
                                y,
                                x0 + 1
                            )

                            edge_src.extend([u, v])
                            edge_dst.extend([v, u])

                        if y + 1 < H:

                            v = node_id(
                                z,
                                y + 1,
                                x0
                            )

                            edge_src.extend([u, v])
                            edge_dst.extend([v, u])

                        if z + 1 < D:

                            v = node_id(
                                z + 1,
                                y,
                                x0
                            )

                            edge_src.extend([u, v])
                            edge_dst.extend([v, u])

            edge_index = torch.tensor(
                [edge_src, edge_dst],
                dtype=torch.long,
                device=feature_map.device
            )

            graphs.append(
                Data(
                    x=nodes,
                    edge_index=edge_index
                )
            )

        return graphs

    def forward(
        self,
        feature_map
    ):

        B, C, D, H, W = feature_map.shape

        graphs = self.build_graph(
            feature_map
        )

        outputs = []

        for graph in graphs:

            x = self.gcn1(
                graph.x,
                graph.edge_index
            )

            x = self.relu(x)

            x = self.gcn2(
                x,
                graph.edge_index
            )

            x = x.t().reshape(
                C,
                D,
                H,
                W
            )

            outputs.append(x)

        refined = torch.stack(
            outputs,
            dim=0
        )

        # Residual refinement
        return feature_map + refined