import numpy as np
from tqdm import tqdm

from .patch_graph import extract_patches
from .edge_builder import build_knn_graph
from .graph_dataset import create_graph


def create_graph_dataset(dataset):

    graphs = []

    for idx in tqdm(range(len(dataset))):

        sample = dataset[idx]

        image = sample["image"].numpy()
        mask = sample["label"].squeeze().numpy()

        nodes, coords, labels = extract_patches(
            image,
            mask,
            patch_size=8
        )

        edge_index = build_knn_graph(
            coords,
            nodes,
            k=8,
            alpha=0.4
        )

        graph = create_graph(
            nodes,
            edge_index,
            labels
        )

        graphs.append(graph)

    return graphs