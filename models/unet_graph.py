import torch
import torch.nn as nn

from models.encoder import Encoder3D
from models.decoder import Decoder3D
from models.graph_refiner import GraphRefiner


class UNetGraph(nn.Module):

    """
    Hybrid UNet + Graph Refinement

    Input:
        B x 4 x 128 x 128 x 128

    Output:
        B x 4 x 128 x 128 x 128
    """

    def __init__(
        self,
        in_channels=4,
        num_classes=4
    ):

        super().__init__()

        self.encoder = Encoder3D(
            in_channels=in_channels
        )

        self.graph = GraphRefiner(
            channels=512
        )

        self.decoder = Decoder3D(
            num_classes=num_classes
        )

    def forward(
        self,
        x
    ):

        # -----------------------
        # Encoder
        # -----------------------

        x1, x2, x3, x4, bottleneck = self.encoder(x)

        # -----------------------
        # Graph Refinement
        # -----------------------

        bottleneck = self.graph(
            bottleneck
        )

        # -----------------------
        # Decoder
        # -----------------------

        out = self.decoder(

            bottleneck,

            x4,

            x3,

            x2,

            x1

        )

        return out


if __name__ == "__main__":

    device = torch.device(

        "cuda"

        if torch.cuda.is_available()

        else "cpu"

    )

    model = UNetGraph().to(device)

    x = torch.randn(

        1,
        4,
        128,
        128,
        128

    ).to(device)

    y = model(x)

    print()

    print("Input :", x.shape)

    print("Output:", y.shape)

    print()

    total = sum(
        p.numel()
        for p in model.parameters()
    )

    trainable = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print("Total Parameters :", total)

    print("Trainable        :", trainable)