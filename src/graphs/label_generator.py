import numpy as np


def patch_label(mask_patch):

    vals, counts = np.unique(
        mask_patch,
        return_counts=True
    )

    # Remove background
    non_zero_mask = vals != 0

    # Pure background patch
    if not np.any(non_zero_mask):
        return 0

    tumor_vals = vals[non_zero_mask]
    tumor_counts = counts[non_zero_mask]

    # Majority tumor class
    return int(
        tumor_vals[
            np.argmax(tumor_counts)
        ]
    )