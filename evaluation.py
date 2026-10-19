from __future__ import annotations

from collections import deque
import math
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import numpy as np


GO_ROOTS = ("GO:0008150", "GO:0005575", "GO:0003674")


def _obo_parent_child(obo_path: str):
    parents = {}
    current = None

    def keep(term):
        if term and term.get("id") and not term.get("obsolete"):
            parents[term["id"]] = term["parents"]

    with open(obo_path, "r", encoding="utf-8") as handle:
        for raw in handle:
            line = raw.strip()
            if line == "[Term]":
                keep(current)
                current = {"parents": set()}
                continue
            if line.startswith("["):
                keep(current)
                current = None
                continue
            if current is None:
                continue
            if line.startswith("id: GO:"):
                current["id"] = line.split()[1]
            elif line == "is_obsolete: true":
                current["obsolete"] = True
            elif line.startswith("is_a: GO:"):
                current["parents"].add(line.split()[1])
            elif line.startswith("relationship: part_of GO:"):
                current["parents"].add(line.split()[2])
    keep(current)

    children = {term: set() for term in parents}
    for child, parent_ids in parents.items():
        for parent in parent_ids:
            if parent in children:
                children[parent].add(child)
    return parents, children


def _reachable(start: str, adjacency: Dict[str, set]) -> set:
    seen = set()
    pending = list(adjacency.get(start, ()))
    while pending:
        term = pending.pop()
        if term not in seen:
            seen.add(term)
            pending.extend(adjacency.get(term, ()))
    return seen


def _fallback_ic(missing: Sequence[str], obo_path: str) -> Dict[str, float]:
    parents, children = _obo_parent_child(obo_path)
    depth = {root: 1 for root in GO_ROOTS if root in children}
    queue = deque(depth)
    while queue:
        parent = queue.popleft()
        for child in children[parent]:
            candidate = depth[parent] + 1
            if child not in depth or candidate < depth[child]:
                depth[child] = candidate
                queue.append(child)
    count = len(parents)
    if count <= 1:
        raise ValueError("GO ontology contains too few terms for IC calculation")
    result = {}
    for term in missing:
        if term not in parents or term not in depth:
            raise ValueError(f"GO term {term} cannot be assigned an IC from {obo_path}")
        ancestors = _reachable(term, parents) | {term}
        descendants = _reachable(term, children)
        descendant_depth_sum = sum(depth[d] for d in descendants)
        value = depth[term] * math.log(len(ancestors)) * (
            1.0 - math.log(descendant_depth_sum + 1.0) / math.log(count)
        )
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"GO term {term} has invalid fallback IC={value}")
        result[term] = value
    return result


def load_ic_weights(go_names: Sequence[str], ic_path: str, obo_path: str) -> Tuple[np.ndarray, Dict[str, object]]:
    ic_by_name = {}
    with open(ic_path, "r", encoding="utf-8") as handle:
        for line_no, raw in enumerate(handle, 1):
            if not raw.strip():
                continue
            parts = raw.split()
            if len(parts) < 3:
                raise ValueError(f"Invalid IC row {line_no} in {ic_path}")
            go_name = parts[0]
            weight = float(parts[2])
            if not math.isfinite(weight) or weight < 0:
                raise ValueError(f"Invalid IC weight for {go_name} in {ic_path}")
            if go_name in ic_by_name:
                raise ValueError(f"Duplicate IC entry for {go_name} in {ic_path}")
            ic_by_name[go_name] = weight
    missing = [name for name in go_names if name not in ic_by_name]
    fallback = _fallback_ic(missing, obo_path) if missing else {}
    ic_by_name.update(fallback)
    weights = np.asarray([ic_by_name[name] for name in go_names], dtype=np.float64)
    return weights, {
        "ic_table": str(Path(ic_path).resolve()),
        "ic_fallback_obo": str(Path(obo_path).resolve()) if missing else None,
        "ic_fallback_values": fallback,
    }


def _micro_metrics(scores: np.ndarray, truth: np.ndarray, weights: np.ndarray) -> Tuple[float, float]:
    positive_weight = weights[truth].sum()
    best_fmax = 0.0
    for i in range(101):
        predicted = scores >= i / 100
        tp = weights[predicted & truth].sum()
        fp = weights[predicted & ~truth].sum()
        fn = weights[~predicted & truth].sum()
        denominator = 2 * tp + fp + fn
        fscore = 2 * tp / denominator if denominator > 0 else 0.0
        best_fmax = max(best_fmax, fscore)
    order = np.argsort(-scores)
    sorted_truth = truth[order]
    sorted_weights = weights[order]
    tp = np.cumsum(sorted_weights * sorted_truth)
    fp = np.cumsum(sorted_weights * ~sorted_truth)
    precision = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) > 0)
    recall = tp / positive_weight if positive_weight > 0 else np.zeros_like(tp)
    aupr = 0.0
    previous_recall = 0.0
    for p, r in zip(precision, recall):
        aupr += p * max(0.0, r - previous_recall)
        previous_recall = r
    return float(best_fmax), float(aupr)


def evaluate_paper_metrics(probs: np.ndarray, labels: np.ndarray,
                           ic_weights: np.ndarray,
                           mask: Optional[np.ndarray] = None) -> Dict[str, float]:
    probs = np.asarray(probs)
    labels = np.asarray(labels)
    weights = np.asarray(ic_weights, dtype=np.float64)
    if probs.ndim != 2 or probs.shape != labels.shape:
        raise ValueError("probs and labels must be equal-shaped 2D matrices")
    if probs.shape[0] == 0:
        raise ValueError("evaluation requires at least one lncRNA")
    if weights.shape != (probs.shape[1],):
        raise ValueError("IC weights must match the GO columns")
    if not np.all(np.isfinite(weights)) or np.any(weights < 0):
        raise ValueError("IC weights must be finite and nonnegative")
    valid = np.ones_like(labels, dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
    if valid.shape != probs.shape or not np.any(valid):
        raise ValueError("mask must match predictions and contain observed pairs")
    scores = probs[valid]
    truth = labels[valid]
    if not np.all(np.isfinite(scores)) or np.any(scores < 0) or np.any(scores > 1):
        raise ValueError("observed probabilities must be finite and within [0, 1]")
    if not np.all((truth == 0) | (truth == 1)):
        raise ValueError("observed labels must be 0 or 1")
    truth = truth.astype(bool)
    pair_weights = np.broadcast_to(weights, probs.shape)[valid]
    fmax, aupr = _micro_metrics(scores, truth, pair_weights)
    u_fmax, u_aupr = _micro_metrics(scores, truth, np.ones(scores.size, dtype=np.float64))
    return {"Fmax": fmax, "AUPR": aupr, "U-Fmax": u_fmax, "U-AUPR": u_aupr}
