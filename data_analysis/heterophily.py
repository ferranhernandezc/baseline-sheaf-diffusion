import torch
import numpy as np


def min_and_max_gains(y: torch.tensor, edge_index: torch.tensor):
    n = y.shape[0]
    classes = list(set(y.tolist()))
    class_probs = [(y == c).sum() / n for c in classes]

    corr_m = torch.zeros((len(classes), len(classes)))

    n_edges = edge_index.shape[1] / 2

    for src, dst in zip(edge_index[0], edge_index[1]):
        if src != dst:
            corr_m[y[src], y[dst]] += 1

    corr_m = corr_m / n_edges

    avg_edge_prob = [
        sum([class_probs[c2] * corr_m[c, c2] for c2 in classes]) for c in classes
    ]

    avg_degree = [n * p_edge for p_edge in avg_edge_prob]

    neigh_distribution = [
        np.array([class_probs[c2] * corr_m[c1, c2] / avg_edge_prob[c1] for c2 in classes])
        for c1 in classes
    ]
    neigh_distribution

    gains = [
        np.linalg.norm(
            np.sqrt(avg_degree[c1]) * neigh_distribution[c1]
            - np.sqrt(avg_degree[c2]) * neigh_distribution[c2]
        )
        for c1 in classes
        for c2 in classes
        if c1 != c2
    ]

    min_gain = (1 / np.sqrt(2)) * np.min(gains)
    max_gain = (1 / np.sqrt(2)) * np.max(gains)
    return min_gain, max_gain


def heterophily_classifier(
    y: torch.tensor, edge_index: torch.tensor, error: float = 1e-4
):
    min_gain, max_gain = min_and_max_gains(y, edge_index)

    if min_gain > 0.2 - error:
        return "Good"
    elif max_gain < 0.2 + error:
        return "Bad"
    return "Mixed"
