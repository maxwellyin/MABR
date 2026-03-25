import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mabr.config import DEFAULT_DATASET_NAME, DEFAULT_MODEL_CHECKPOINT, DEFAULT_NUM_LABELS
from mabr.losses import debiased_focal_loss, focal_loss
from mabr.models import BiasDetector, DomainClassifier, ReverseLayerF

# Legacy compatibility constants used by the original scripts.
MODEL_CHECKPOINT = DEFAULT_MODEL_CHECKPOINT
DATA_NAME = DEFAULT_DATASET_NAME
NUM_LABELS = DEFAULT_NUM_LABELS


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {"accuracy": accuracy_score(labels, predictions)}


def adjust_logits_for_hans_sum(logits):
    entailment_logits = logits[:, 0]
    non_entailment_logits = logits[:, 1] + logits[:, 2]
    return np.stack([entailment_logits, non_entailment_logits], axis=1)


def adjust_logits_for_hans_max(logits):
    entailment_logits = logits[:, 0]
    non_entailment_logits = np.maximum(logits[:, 1], logits[:, 2])
    return np.stack([entailment_logits, non_entailment_logits], axis=1)


def compute_metrics_hans(eval_pred):
    logits, labels = eval_pred
    adjusted_logits_sum = adjust_logits_for_hans_sum(logits)
    predictions_sum = np.argmax(adjusted_logits_sum, axis=-1)
    adjusted_logits_max = adjust_logits_for_hans_max(logits)
    predictions_max = np.argmax(adjusted_logits_max, axis=-1)
    return {
        "accuracy_sum": accuracy_score(labels, predictions_sum),
        "accuracy_max": accuracy_score(labels, predictions_max),
    }
