from __future__ import annotations

import math
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np



def align_window_scores_to_point_labels(
    scores: np.ndarray,
    dataset: Any,
    window_size: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Align one score per sliding window to point-wise labels."""
    scores = np.asarray(scores).reshape(-1)
    labels_parts: List[np.ndarray] = []
    score_parts: List[np.ndarray] = []
    offset = 0

    for entity in dataset.entities:
        labels = np.asarray(entity.labels).reshape(-1).astype(np.int64)
        n_windows = max(entity.n_time - window_size + 1, 0)
        entity_scores = scores[offset : offset + n_windows]
        offset += n_windows
        if n_windows == 0:
            continue

        aligned_scores = np.concatenate(
            [np.zeros(window_size - 1, dtype=float), entity_scores.astype(float)]
        )
        aligned_scores = aligned_scores[: len(labels)]
        labels_parts.append(labels[: len(aligned_scores)])
        score_parts.append(aligned_scores)

    if offset != len(scores):
        raise ValueError(
            f"Score length mismatch: consumed {offset} window scores, got {len(scores)}."
        )
    if not labels_parts:
        return np.array([], dtype=np.int64), np.array([], dtype=float)

    return np.concatenate(labels_parts), np.concatenate(score_parts)


def compute_metrics(
    labels: np.ndarray,
    scores: np.ndarray,
    metric_window: int,
) -> Dict[str, Any]:
    labels, scores = _clean_labels_scores(labels, scores)
    vus_roc, vus_pr = _vus(
        labels=labels,
        scores=scores,
        max_window=metric_window,
        n_cutoffs=250,
    )
    return json_ready({
        "VUS-ROC": vus_roc,
        "VUS-PR": vus_pr,
    })


def json_ready(metrics: Dict[str, Any]) -> Dict[str, Any]:
    ready: Dict[str, Any] = {}
    for key, value in metrics.items():
        if isinstance(value, np.generic):
            value = value.item()
        if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
            ready[key] = None
        else:
            ready[key] = value
    return ready


def _clean_labels_scores(labels: np.ndarray, scores: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    labels = np.asarray(labels).reshape(-1)
    scores = np.asarray(scores).reshape(-1).astype(float)
    if len(labels) != len(scores):
        raise ValueError(f"Label/score length mismatch: {len(labels)} vs {len(scores)}.")
    labels = (labels > 0).astype(np.int64)
    mask = np.isfinite(scores)
    return labels[mask], scores[mask]


def _vus(
    labels: np.ndarray,
    scores: np.ndarray,
    max_window: int,
    n_cutoffs: int,
) -> Tuple[float, float]:
    labels = np.asarray(labels).reshape(-1).astype(np.int64)
    scores = np.asarray(scores).reshape(-1).astype(float)
    original_ranges = _binary_ranges(labels)
    if len(scores) == 0 or not original_ranges or len(np.unique(labels)) < 2:
        return float("nan"), float("nan")

    max_window = max(int(max_window), 0)
    max_window_ranges = _expanded_ranges(len(labels), original_ranges, max_window)
    if not max_window_ranges:
        return float("nan"), float("nan")

    cutoffs = _score_cutoffs(scores, n_cutoffs)
    pred_by_cutoff = [scores >= cutoff for cutoff in cutoffs]
    n_pred_by_cutoff = np.asarray([pred.sum() for pred in pred_by_cutoff], dtype=float)
    p_original = float(labels.sum())

    roc_values: List[float] = []
    pr_values: List[float] = []

    for window in range(max_window + 1):
        extended_labels = _extend_labels(labels, original_ranges, window)
        window_ranges = _expanded_ranges(len(labels), original_ranges, window)
        if not window_ranges:
            continue

        tf_points: List[Tuple[float, float]] = [(0.0, 0.0)]
        precision_points: List[float] = [1.0]
        for pred, n_pred in zip(pred_by_cutoff, n_pred_by_cutoff):
            weighted_labels = extended_labels.copy()
            existence = 0
            for start, end in window_ranges:
                segment_pred = pred[start : end + 1]
                weighted_labels[start : end + 1] *= segment_pred
                if segment_pred.any():
                    existence += 1
            for start, end in original_ranges:
                weighted_labels[start : end + 1] = 1.0

            tp = 0.0
            n_labels = 0.0
            for start, end in max_window_ranges:
                tp += float(np.dot(weighted_labels[start : end + 1], pred[start : end + 1]))
                n_labels += float(weighted_labels[start : end + 1].sum())

            p_new = max((p_original + n_labels) / 2.0, 1e-12)
            recall = min(tp / p_new, 1.0)
            existence_ratio = existence / len(window_ranges)
            tpr = recall * existence_ratio

            fp = max(float(n_pred) - tp, 0.0)
            n_new = max(float(len(labels)) - p_new, 1e-12)
            fpr = fp / n_new
            precision = 1.0 if n_pred <= 0 else tp / float(n_pred)

            tf_points.append((float(np.clip(tpr, 0.0, 1.0)), float(np.clip(fpr, 0.0, 1.0))))
            precision_points.append(float(np.clip(precision, 0.0, 1.0)))

        tf_points.append((1.0, 1.0))
        tf = np.asarray(tf_points, dtype=float)
        precision = np.asarray(precision_points, dtype=float)

        width = tf[1:, 1] - tf[:-1, 1]
        height = (tf[1:, 0] + tf[:-1, 0]) / 2.0
        roc_values.append(_clip_auc(float(np.dot(width, height))))

        width_pr = tf[1:-1, 0] - tf[:-2, 0]
        height_pr = precision[1:]
        pr_values.append(_clip_auc(float(np.dot(width_pr, height_pr))))

    if not roc_values or not pr_values:
        return float("nan"), float("nan")
    return float(np.mean(roc_values)), float(np.mean(pr_values))


def _binary_ranges(values: np.ndarray) -> List[Tuple[int, int]]:
    binary = np.asarray(values).reshape(-1) > 0
    if len(binary) == 0:
        return []
    starts = np.flatnonzero(binary & np.r_[True, ~binary[:-1]])
    ends = np.flatnonzero(binary & np.r_[~binary[1:], True])
    return [(int(start), int(end)) for start, end in zip(starts, ends)]


def _extend_labels(
    labels: np.ndarray,
    ranges: Sequence[Tuple[int, int]],
    window: int,
) -> np.ndarray:
    labels = np.asarray(labels).reshape(-1).astype(float)
    if window <= 0:
        return labels

    extended = labels.copy()
    n = len(labels)
    half_window = window // 2
    for start, end in ranges:
        right = np.arange(end + 1, min(end + half_window + 1, n))
        if len(right):
            extended[right] += np.sqrt(1.0 - (right - end) / window)
        left = np.arange(max(start - half_window, 0), start)
        if len(left):
            extended[left] += np.sqrt(1.0 - (start - left) / window)
    return np.minimum(np.ones(n, dtype=float), extended)


def _expanded_ranges(
    n_points: int,
    ranges: Sequence[Tuple[int, int]],
    window: int,
) -> List[Tuple[int, int]]:
    if not ranges:
        return []
    half_window = max(int(window), 0) // 2
    expanded: List[Tuple[int, int]] = []
    current_start = max(ranges[0][0] - half_window, 0)
    current_end = min(ranges[0][1] + half_window, n_points - 1)
    for start, end in ranges[1:]:
        next_start = max(start - half_window, 0)
        next_end = min(end + half_window, n_points - 1)
        if current_end < next_start:
            expanded.append((current_start, current_end))
            current_start, current_end = next_start, next_end
        else:
            current_end = max(current_end, next_end)
    expanded.append((current_start, current_end))
    return expanded


def _score_cutoffs(scores: np.ndarray, n_cutoffs: int) -> np.ndarray:
    scores_sorted = np.sort(np.asarray(scores).reshape(-1).astype(float))[::-1]
    if len(scores_sorted) == 0:
        return np.array([], dtype=float)
    n_cutoffs = min(max(int(n_cutoffs), 1), len(scores_sorted))
    indices = np.linspace(0, len(scores_sorted) - 1, n_cutoffs).astype(int)
    return scores_sorted[indices]


def _clip_auc(value: float) -> float:
    if not math.isfinite(value):
        return float("nan")
    return float(np.clip(value, 0.0, 1.0))
