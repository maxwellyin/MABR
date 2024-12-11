# %%
import os
from torch.utils.data import DataLoader
import torch
import torch.nn.functional as F
from tqdm import tqdm
from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding, pipeline
from utils.util import focal_loss, MODEL_CHECKPOINT, NUM_LABELS, DATA_NAME
import wandb
import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score
import numpy as np
from scipy.stats import entropy

# %%
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
wandb.init()
data_name = DATA_NAME
model_name = MODEL_CHECKPOINT.split("/")[-1]
data = f"../data/{data_name}"
checkpoint_dir = f"./checkpoint/{model_name}-{data_name}/dann/"
test_split = "test"
# test_split = "balanced_test"

dataset = load_from_disk(data)
dataset = dataset.rename_column("gender", "race")

tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT, model_max_length=128)
# %%
def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True)

tokenized_datasets = dataset.map(tokenize_function, load_from_cache_file=True).remove_columns(['text'])

test_dataset = tokenized_datasets[test_split]
# %%
tokenized_datasets.set_format("torch")

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

validation_loader = DataLoader(
    test_dataset,
    batch_size=16,
    shuffle=False,  # Usually, we don't shuffle the validation set.
    collate_fn=data_collator
)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def validate_model(model, validation_loader, device, gamma=2.0):
    model.eval()
    total_loss, total_correct, total_samples = 0, 0, 0
    all_labels = []
    all_predictions = []
    with torch.no_grad():
        for batch in tqdm(validation_loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'], output_hidden_states=True)

            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1)  # Use argmax to get the index of the highest logit
            loss = focal_loss(logits, batch['labels'], gamma)
            total_loss += loss.item()
            total_correct += (predictions == batch['labels']).sum().item()
            total_samples += batch['labels'].size(0)
            all_labels.extend(batch['labels'].cpu().numpy())
            all_predictions.extend(predictions.cpu().numpy())
    
    avg_loss = total_loss / len(validation_loader)
    accuracy = total_correct / total_samples
    wandb.log({"val_loss": avg_loss, "val_accuracy": accuracy})
    print(f"Validation - Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")

    return np.array(all_labels), np.array(all_predictions)

def calculate_tpr_fpr(df, group_col, label_col, pred_col, num_labels):
    metrics = {}
    for group in df[group_col].unique():
        group_df = df[df[group_col] == group]
        cm = confusion_matrix(group_df[label_col], group_df[pred_col], labels=range(num_labels))
        TPR = []
        FPR = []
        for i in range(num_labels):
            TP = cm[i, i]
            FP = cm[:, i].sum() - TP
            FN = cm[i, :].sum() - TP
            TN = cm.sum() - TP - FP - FN
            TPR.append(TP / (TP + FN) if TP + FN > 0 else 0)
            FPR.append(FP / (FP + TN) if FP + TN > 0 else 0)
        metrics[group] = {'TPR': TPR, 'FPR': FPR}
    return metrics

def calculate_gap(metrics_dict):
    groups = list(metrics_dict.keys())
    tpr_gaps = []
    fpr_gaps = []
    for i in range(len(metrics_dict[groups[0]]['TPR'])):
        tpr_gap = abs(metrics_dict[groups[0]]['TPR'][i] - metrics_dict[groups[1]]['TPR'][i])
        fpr_gap = abs(metrics_dict[groups[0]]['FPR'][i] - metrics_dict[groups[1]]['FPR'][i])
        tpr_gaps.append(tpr_gap)
        fpr_gaps.append(fpr_gap)
    return np.mean(tpr_gaps), np.mean(fpr_gaps)

def calculate_rms_tpr_gaps(df, label_col, pred_col, race_col, num_labels):
    gap_squares = []
    metrics = calculate_tpr_fpr(df, race_col, label_col, pred_col, num_labels)
    for i in range(num_labels):
        tpr_gap = abs(metrics[list(metrics.keys())[0]]['TPR'][i] - metrics[list(metrics.keys())[1]]['TPR'][i])
        gap_squares.append(tpr_gap**2)
    rms_tpr_gap = np.sqrt(np.mean(gap_squares))
    return rms_tpr_gap

def kl_divergence(p, q, epsilon=1e-10):
    p = p + epsilon
    q = q + epsilon
    return entropy(p, q, base=np.e)

def calculate_distributions(predictions, attribute, num_classes):
    p_r = np.zeros(num_classes)
    p_r_given_z = {z: np.zeros(num_classes) for z in np.unique(attribute)}

    for i in range(num_classes):
        p_r[i] = np.mean(predictions == i)
        for z in p_r_given_z:
            p_r_given_z[z][i] = np.mean(predictions[attribute == z] == i)

    return p_r, p_r_given_z

def independence(predictions, attribute, num_classes):
    p_r, p_r_given_z = calculate_distributions(predictions, attribute, num_classes)
    kl_sum = sum(kl_divergence(p_r, p_r_given_z[z]) for z in p_r_given_z)
    return kl_sum

def calculate_separation(predictions, labels, attribute, num_classes):
    p_r_given_y = {y: np.zeros(num_classes) for y in np.unique(labels)}
    p_r_given_y_z = {y: {z: np.zeros(num_classes) for z in np.unique(attribute)} for y in np.unique(labels)}

    for y in p_r_given_y:
        for i in range(num_classes):
            p_r_given_y[y][i] = np.mean(predictions[labels == y] == i)
            for z in p_r_given_y_z[y]:
                p_r_given_y_z[y][z][i] = np.mean(predictions[(labels == y) & (attribute == z)] == i)

    kl_sum = 0
    for y in p_r_given_y:
        for z in p_r_given_y_z[y]:
            kl_sum += kl_divergence(p_r_given_y[y], p_r_given_y_z[y][z])

    return kl_sum

def calculate_sufficiency(predictions, labels, attribute, num_classes):
    p_y_given_r = {r: np.zeros(num_classes) for r in np.unique(predictions)}
    p_y_given_r_z = {r: {z: np.zeros(num_classes) for z in np.unique(attribute)} for r in np.unique(predictions)}

    for r in p_y_given_r:
        for i in range(num_classes):
            p_y_given_r[r][i] = np.mean(labels[predictions == r] == i)
            for z in p_y_given_r_z[r]:
                p_y_given_r_z[r][z][i] = np.mean(labels[(predictions == r) & (attribute == z)] == i)

    kl_sum = 0
    for r in p_y_given_r:
        for z in p_y_given_r_z[r]:
            kl_sum += kl_divergence(p_y_given_r[r], p_y_given_r_z[r][z])

    return kl_sum
# %%
for checkpoint in os.listdir(checkpoint_dir):
    checkpoint_path = os.path.join(checkpoint_dir, checkpoint)
    if os.path.isdir(checkpoint_path) and (checkpoint.startswith('checkpoint') or checkpoint.startswith('epoch')):
        print(f"Evaluating checkpoint: {checkpoint_path}")
        
        # Load the model from the checkpoint
        model = AutoModelForSequenceClassification.from_pretrained(checkpoint_path, num_labels=NUM_LABELS)
        model.to(device)
        
        # Validate the model
        labels, predictions = validate_model(model, validation_loader, device)
        
        # Fairness evaluation
        df = pd.DataFrame({'label': labels, 'prediction': predictions, 'race': test_dataset['race']})
        
        # Calculate metrics
        total_accuracy = accuracy_score(df['label'], df['prediction'])
        wandb.log({"total_accuracy": total_accuracy})
        
        # Calculate TPR and FPR for each demographic group
        metrics = calculate_tpr_fpr(df, 'race', 'label', 'prediction', num_labels=NUM_LABELS)
        tpr_gap, fpr_gap = calculate_gap(metrics)
        wandb.log({"tpr_gap": tpr_gap, "fpr_gap": fpr_gap})

        # Calculate RMS TPR Gap
        rms_tpr_gap = calculate_rms_tpr_gaps(df, 'label', 'prediction', 'race', num_labels=NUM_LABELS)
        wandb.log({"rms_tpr_gap": rms_tpr_gap})

        # Calculate Independence
        independence_score = independence(df['prediction'].astype(int), df['race'].astype(int), NUM_LABELS)
        wandb.log({"independence_score": independence_score})

        # Calculate Separation
        separation_score = calculate_separation(df['prediction'].astype(int), df['label'].astype(int), df['race'].astype(int), NUM_LABELS)
        wandb.log({"separation_score": separation_score})

        # Calculate Sufficiency
        sufficiency_score = calculate_sufficiency(df['prediction'].astype(int), df['label'].astype(int), df['race'].astype(int), NUM_LABELS)
        wandb.log({"sufficiency_score": sufficiency_score})

        print(f"Checkpoint: {checkpoint_path}")
        print(f"Total Accuracy: {total_accuracy}")
        print(f"TPR Gap: {tpr_gap}, FPR Gap: {fpr_gap}")
        print(f"RMS TPR Gap: {rms_tpr_gap}")
        print(f"Independence: {independence_score}")
        print(f"Separation: {separation_score}")
        print(f"Sufficiency: {sufficiency_score}")

wandb.finish()
# %%
