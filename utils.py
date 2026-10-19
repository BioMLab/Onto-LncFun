from collections import deque
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


def read_names(path):
    names = [row.split()[-1] for row in Path(path).read_text(encoding='utf-8-sig').splitlines() if row.strip()]
    if len(set(names)) != len(names):
        raise ValueError(f'Duplicate identifiers in {path}')
    return names


def load_features(path, names):
    values = np.load(path, allow_pickle=False)
    if values.ndim != 2 or values.shape[0] != len(names) or not np.isfinite(values).all():
        raise ValueError(f'Invalid or misaligned features in {path}')
    return torch.as_tensor(values, dtype=torch.float32)


def read_samples(path, names, go_names):
    lnc_index = {name: i for i, name in enumerate(names)}
    go_index = {name: i for i, name in enumerate(go_names)}
    result = {}
    with Path(path).open(encoding='utf-8-sig') as handle:
        for row in handle:
            parts = row.split()
            if not parts or parts[0] == 'lncRNAname':
                continue
            if len(parts) != 3 or parts[0] not in lnc_index or parts[1] not in go_index:
                raise ValueError(f'Invalid training sample: {row.strip()}')
            label = int(parts[2])
            if label not in (0, 1):
                raise ValueError(f'Invalid label: {row.strip()}')
            pair = (lnc_index[parts[0]], go_index[parts[1]])
            if pair in result and result[pair] != label:
                raise ValueError(f'Conflicting labels: {row.strip()}')
            result[pair] = label
    if not result:
        raise ValueError('No training samples')
    labels = np.zeros((len(names), len(go_names)), dtype=np.float32)
    mask = np.zeros_like(labels, dtype=bool)
    for (i, j), value in result.items():
        labels[i, j] = value
        mask[i, j] = True
    if not mask.any(axis=1).all():
        raise ValueError('Training index includes an unlabeled lncRNA')
    return labels, mask


def read_test_labels(path, names, go_names):
    lnc_index = {name: i for i, name in enumerate(names)}
    go_index = {name: i for i, name in enumerate(go_names)}
    labels = np.zeros((len(names), len(go_names)), dtype=np.float32)
    covered = np.zeros(len(go_names), dtype=bool)
    with Path(path).open(encoding='utf-8-sig') as handle:
        for row in handle:
            parts = row.split()
            if not parts:
                continue
            if len(parts) != 2 or parts[0] not in lnc_index:
                raise ValueError(f'Invalid test annotation: {row.strip()}')
            if parts[1] not in go_index:
                continue
            j = go_index[parts[1]]
            labels[lnc_index[parts[0]], j] = 1
            covered[j] = True
    if not covered.any():
        raise ValueError('Test annotations have no GO terms in the model')
    return labels, np.broadcast_to(covered, labels.shape)


def depth_cost(num_go, edges, roots, device):
    children = [[] for _ in range(num_go)]
    for parent, child in edges.detach().cpu().t().tolist():
        children[parent].append(child)
    depth = [None] * num_go
    queue = deque(roots)
    for root in roots:
        depth[root] = 0
    while queue:
        parent = queue.popleft()
        for child in children[parent]:
            candidate = depth[parent] + 1
            if depth[child] is None or candidate < depth[child]:
                depth[child] = candidate
                queue.append(child)
    finite = [value for value in depth if value is not None]
    if finite:
        fallback = max(finite) + 1
        depth = [fallback if value is None else value for value in depth]
    else:
        depth = [0] * num_go
    depth = torch.tensor(depth, dtype=torch.float32, device=device)
    cost = 1.0 + depth.max().clamp(min=1.0) - depth
    return cost / cost.mean().clamp(min=1.0)


def hierarchical_loss(logits, labels, mask, costs, edges):
    probabilities = torch.sigmoid(logits)
    base = ((labels * (1.0 - probabilities) + (1.0 - labels) * probabilities) * costs.unsqueeze(0) * mask).sum() / mask.sum().clamp(min=1)
    if edges.numel():
        parents, children = edges
        penalty = (F.relu(probabilities[:, children] - probabilities[:, parents]) * costs[children].unsqueeze(0)).mean()
        return base + 0.1 * penalty
    return base


def save_predictions(path, names, go_names, probabilities, labels, mask):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as handle:
        handle.write('lncRNAname\tGOname\ty_prob\ty_true\n')
        rows, columns = np.nonzero(mask)
        for i, j in zip(rows, columns):
            handle.write(f'{names[i]}\t{go_names[j]}\t{probabilities[i,j]:.6f}\t{int(labels[i,j])}\n')
