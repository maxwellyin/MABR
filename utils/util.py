import numpy as np
from sklearn.metrics import accuracy_score
import torch.nn as nn
import torch
import torch.nn.functional as F

# "bert-base-uncased" "roberta-base"
MODEL_CHECKPOINT = "roberta-base"
DATA_NAME = "biosbias"
NUM_LABELS = 28

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
    adjusted_logits = np.stack([entailment_logits, non_entailment_logits], axis=1)
    return adjusted_logits

# def compute_metrics_hans(eval_pred, use_sum=False):
#     logits, labels = eval_pred
#     # Choose between sum or max based on the use_sum flag
#     if use_sum:
#         adjusted_logits = adjust_logits_for_hans_sum(logits)
#     else:
#         adjusted_logits = adjust_logits_for_hans_max(logits)
#     predictions = np.argmax(adjusted_logits, axis=-1)
#     return {"accuracy": accuracy_score(labels, predictions)}

def compute_metrics_hans(eval_pred):
    logits, labels = eval_pred
    adjusted_logits_sum = adjust_logits_for_hans_sum(logits)
    predictions_sum = np.argmax(adjusted_logits_sum, axis=-1)
    accuracy_sum = accuracy_score(labels, predictions_sum)
    
    adjusted_logits_max = adjust_logits_for_hans_max(logits)
    predictions_max = np.argmax(adjusted_logits_max, axis=-1)
    accuracy_max = accuracy_score(labels, predictions_max)
    
    return {
        "accuracy_sum": accuracy_sum,
        "accuracy_max": accuracy_max
    }

class BiasDetector(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.fc = nn.Linear(input_dim, 1)  # Output 1 as you want to predict binary correct/incorrect

    def forward(self, x):
        return torch.sigmoid(self.fc(x)) 
    
class DomainClassifier(nn.Module):
    def __init__(self, input_dim):
        super(DomainClassifier, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 384),
            nn.ReLU(),
            nn.Linear(384, 192),
            nn.ReLU(),
            nn.Linear(192, 2),
            nn.LogSoftmax(dim=1)
        )
    
    def forward(self, x):
        return self.model(x)
    
# Gradient Reversal Layer
class ReverseLayerF(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None

def focal_loss(inputs, targets, gamma):
    # Apply softmax to inputs
    log_probs = F.log_softmax(inputs, dim=-1)
    probs = torch.exp(log_probs)

    # Gather the log probabilities for the correct class
    targets = targets.view(-1, 1)
    log_probs = log_probs.gather(1, targets)
    probs = probs.gather(1, targets)

    # Compute the focal loss
    focal_loss = -((1 - probs) ** gamma) * log_probs
    return focal_loss.mean()


def debiased_focal_loss(inputs, targets, gamma, bias_preds):
    # Compute log probabilities using softmax
    log_probs = F.log_softmax(inputs, dim=-1)
    
    # Gather the log probabilities and probabilities for the actual target classes
    gather_indices = targets.view(-1, 1)
    log_probs = log_probs.gather(1, gather_indices)
    probs = torch.exp(log_probs)

    # Compute the focal loss component
    focal_loss = -((1 - probs) ** gamma) * log_probs
    
    # Re-weight the focal loss using the exponential of the negative bias predictions
    debiased_focal_loss = torch.exp(-bias_preds) * focal_loss

    # Return the mean of the debiased focal loss
    return debiased_focal_loss.mean()