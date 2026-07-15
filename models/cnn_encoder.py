import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNEncoder(nn.Module):

    def __init__(self):

        super().__init__()

        self.features = nn.Sequential(

            nn.Conv3d(
                4,
                16,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm3d(16),
            nn.ReLU(),

            nn.MaxPool3d(2),

            nn.Conv3d(
                16,
                32,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm3d(32),
            nn.ReLU(),

            nn.AdaptiveAvgPool3d(1)
        )

        self.fc = nn.Linear(
            32,
            64
        )

    def forward(self, x):

        x = self.features(x)

        x = x.flatten(1)

        x = self.fc(x)

        return x