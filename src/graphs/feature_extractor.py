import numpy as np
import torch

from models.cnn_encoder import CNNEncoder


device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

cnn = CNNEncoder().to(device)
cnn.eval()


def patch_features(image_patch):

    patch_tensor = torch.tensor(
        image_patch,
        dtype=torch.float32
    )

    patch_tensor = patch_tensor.unsqueeze(0).to(device)

    with torch.no_grad():

        features = cnn(
            patch_tensor
        )

    features = features.squeeze().cpu().numpy()

    return features.astype(np.float32)