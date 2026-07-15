import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):

    def __init__(
        self,
        smooth=1.0
    ):

        super().__init__()

        self.smooth = smooth

    def forward(
        self,
        pred,
        target
    ):

        # pred : [B,C,D,H,W]
        # target : [B,D,H,W] or [B,1,D,H,W]

        if target.dim() == 5:
            target = target.squeeze(1)

        num_classes = pred.shape[1]

        pred = F.softmax(
            pred,
            dim=1
        )

        target_onehot = F.one_hot(

            target.long(),

            num_classes=num_classes

        )

        target_onehot = target_onehot.permute(

            0,
            4,
            1,
            2,
            3

        ).float()

        pred = pred.contiguous().view(

            pred.shape[0],
            num_classes,
            -1

        )

        target_onehot = target_onehot.contiguous().view(

            target_onehot.shape[0],
            num_classes,
            -1

        )

        intersection = (

            pred *
            target_onehot

        ).sum(-1)

        denominator = (

            pred.sum(-1) +
            target_onehot.sum(-1)

        )

        dice = (

            2 * intersection +
            self.smooth

        ) / (

            denominator +
            self.smooth

        )

        loss = 1 - dice.mean()

        return loss


class CombinedLoss(nn.Module):

    def __init__(
        self,
        dice_weight=0.7,
        ce_weight=0.3
    ):

        super().__init__()

        self.dice = DiceLoss()

        self.ce = nn.CrossEntropyLoss()

        self.dw = dice_weight

        self.cw = ce_weight

    def forward(
        self,
        pred,
        target
    ):

        if target.dim() == 5:

            target = target.squeeze(1)

        target = target.long()

        dice_loss = self.dice(

            pred,

            target

        )

        ce_loss = self.ce(

            pred,

            target

        )

        total_loss = (

            self.dw * dice_loss +

            self.cw * ce_loss

        )

        return total_loss


if __name__ == "__main__":

    pred = torch.randn(

        2,
        4,
        128,
        128,
        128

    )

    target = torch.randint(

        0,
        4,
        (
            2,
            128,
            128,
            128
        )

    )

    criterion = CombinedLoss()

    loss = criterion(

        pred,

        target

    )

    print(loss)