from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import entropy
from sklearn.metrics import accuracy_score, confusion_matrix


def compute_accuracy_metrics(eval_pred: tuple[np.ndarray, np.ndarray]) -> dict[str, float]:
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {"accuracy": accuracy_score(labels, predictions)}


def calculate_tpr_fpr(
    df: pd.DataFrame,
    group_col: str,
    label_col: str,
    pred_col: str,
    num_labels: int,
) -> dict[object, dict[str, list[float]]]:
    metrics: dict[object, dict[str, list[float]]] = {}
    for group in df[group_col].unique():
        group_df = df[df[group_col] == group]
        cm = confusion_matrix(group_df[label_col], group_df[pred_col], labels=range(num_labels))
        tpr: list[float] = []
        fpr: list[float] = []
        for i in range(num_labels):
            tp = cm[i, i]
            fp = cm[:, i].sum() - tp
            fn = cm[i, :].sum() - tp
            tn = cm.sum() - tp - fp - fn
            tpr.append(tp / (tp + fn) if tp + fn > 0 else 0.0)
            fpr.append(fp / (fp + tn) if fp + tn > 0 else 0.0)
        metrics[group] = {"TPR": tpr, "FPR": fpr}
    return metrics


def calculate_gap(metrics_dict: dict[object, dict[str, list[float]]]) -> tuple[float, float]:
    groups = list(metrics_dict.keys())
    if len(groups) < 2:
        return 0.0, 0.0
    tpr_gaps = []
    fpr_gaps = []
    for i in range(len(metrics_dict[groups[0]]["TPR"])):
        tpr_gaps.append(abs(metrics_dict[groups[0]]["TPR"][i] - metrics_dict[groups[1]]["TPR"][i]))
        fpr_gaps.append(abs(metrics_dict[groups[0]]["FPR"][i] - metrics_dict[groups[1]]["FPR"][i]))
    return float(np.mean(tpr_gaps)), float(np.mean(fpr_gaps))


def calculate_rms_tpr_gap(
    df: pd.DataFrame,
    label_col: str,
    pred_col: str,
    group_col: str,
    num_labels: int,
) -> float:
    gap_squares = []
    metrics = calculate_tpr_fpr(df, group_col, label_col, pred_col, num_labels)
    groups = list(metrics.keys())
    if len(groups) < 2:
        return 0.0
    for i in range(num_labels):
        tpr_gap = abs(metrics[groups[0]]["TPR"][i] - metrics[groups[1]]["TPR"][i])
        gap_squares.append(tpr_gap**2)
    return float(np.sqrt(np.mean(gap_squares)))


def kl_divergence(p: np.ndarray, q: np.ndarray, epsilon: float = 1e-10) -> float:
    return float(entropy(p + epsilon, q + epsilon, base=np.e))


def calculate_distributions(
    predictions: np.ndarray,
    attribute: np.ndarray,
    num_classes: int,
) -> tuple[np.ndarray, dict[object, np.ndarray]]:
    p_r = np.zeros(num_classes)
    p_r_given_z = {z: np.zeros(num_classes) for z in np.unique(attribute)}
    for i in range(num_classes):
        p_r[i] = np.mean(predictions == i)
        for z in p_r_given_z:
            p_r_given_z[z][i] = np.mean(predictions[attribute == z] == i)
    return p_r, p_r_given_z


def independence(predictions: np.ndarray, attribute: np.ndarray, num_classes: int) -> float:
    p_r, p_r_given_z = calculate_distributions(predictions, attribute, num_classes)
    return float(sum(kl_divergence(p_r, p_r_given_z[z]) for z in p_r_given_z))


def separation(
    predictions: np.ndarray,
    labels: np.ndarray,
    attribute: np.ndarray,
    num_classes: int,
) -> float:
    p_r_given_y = {y: np.zeros(num_classes) for y in np.unique(labels)}
    p_r_given_y_z = {
        y: {z: np.zeros(num_classes) for z in np.unique(attribute)}
        for y in np.unique(labels)
    }
    for y in p_r_given_y:
        for i in range(num_classes):
            p_r_given_y[y][i] = np.mean(predictions[labels == y] == i)
            for z in p_r_given_y_z[y]:
                p_r_given_y_z[y][z][i] = np.mean(predictions[(labels == y) & (attribute == z)] == i)
    return float(
        sum(
            kl_divergence(p_r_given_y[y], p_r_given_y_z[y][z])
            for y in p_r_given_y
            for z in p_r_given_y_z[y]
        )
    )


def sufficiency(
    predictions: np.ndarray,
    labels: np.ndarray,
    attribute: np.ndarray,
    num_classes: int,
) -> float:
    p_y_given_r = {r: np.zeros(num_classes) for r in np.unique(predictions)}
    p_y_given_r_z = {
        r: {z: np.zeros(num_classes) for z in np.unique(attribute)}
        for r in np.unique(predictions)
    }
    for r in p_y_given_r:
        for i in range(num_classes):
            p_y_given_r[r][i] = np.mean(labels[predictions == r] == i)
            for z in p_y_given_r_z[r]:
                p_y_given_r_z[r][z][i] = np.mean(labels[(predictions == r) & (attribute == z)] == i)
    return float(
        sum(
            kl_divergence(p_y_given_r[r], p_y_given_r_z[r][z])
            for r in p_y_given_r
            for z in p_y_given_r_z[r]
        )
    )
