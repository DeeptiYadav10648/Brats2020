import torch
import numpy as np
from sklearn.neighbors import NearestNeighbors


def build_knn_graph(
    coords,
    features,
    k=8,
    alpha=0.4
):
    """
    alpha:
        1.0 -> only spatial distance
        0.0 -> only feature distance
    """

    coords = coords.astype(np.float32)
    features = features.astype(np.float32)

    coords = coords / (coords.std(axis=0) + 1e-8)
    features = features / (features.std(axis=0) + 1e-8)

    combined = np.concatenate(
        [
            alpha * coords,
            (1 - alpha) * features
        ],
        axis=1
    )

    knn = NearestNeighbors(
        n_neighbors=k + 1
    )

    knn.fit(combined)

    indices = knn.kneighbors(
        return_distance=False
    )

    edge_list = []

    for i in range(len(indices)):

        for j in indices[i][1:]:

            edge_list.append([i, j])
            edge_list.append([j, i])

    edge_index = torch.tensor(
        edge_list,
        dtype=torch.long
    ).t()

    return edge_index