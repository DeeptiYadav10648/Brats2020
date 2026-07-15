import numpy as np

from .feature_extractor import patch_features
from .label_generator import patch_label


def extract_patches(
    volume,
    mask,
    patch_size=8
):

    nodes = []
    coords = []
    labels = []

    _, H, W, D = volume.shape

    for x in range(
        0,
        H - patch_size,
        patch_size
    ):
        for y in range(
            0,
            W - patch_size,
            patch_size
        ):
            for z in range(
                0,
                D - patch_size,
                patch_size
            ):

                patch = volume[
                    :,
                    x:x+patch_size,
                    y:y+patch_size,
                    z:z+patch_size
                ]

                feat = patch_features(
                    patch
                )

                nodes.append(feat)

                coords.append(
                    [x, y, z]
                )

                mask_patch = mask[
                    x:x+patch_size,
                    y:y+patch_size,
                    z:z+patch_size
                ]

                labels.append(
                    patch_label(mask_patch)
                )

    return (
        np.array(nodes),
        np.array(coords),
        np.array(labels)
    )