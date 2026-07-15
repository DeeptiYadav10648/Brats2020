import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):

    def __init__(
        self,
        smooth=1e-5
    ):
        super().__init__()

        self.smooth = smooth

    def forward(
        self,
        logits,
        targets
    ):

        num_classes = logits.shape[1]

        probs = F.softmax(
            logits,
            dim=1
        )

        targets_one_hot = F.one_hot(
            targets,
            num_classes
        ).float()

        dims = (0,)

        intersection = (
            probs * targets_one_hot
        ).sum(dims)

        union = (
            probs + targets_one_hot
        ).sum(dims)

        dice = (
            2 * intersection + self.smooth
        ) / (
            union + self.smooth
        )

        return 1 - dice.mean()