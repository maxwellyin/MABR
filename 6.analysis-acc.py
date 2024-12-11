# %%
import os
import torch
import shutil
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from tqdm import tqdm
from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding
from utils.util import ReverseLayerF, MODEL_CHECKPOINT, BiasDetector, DomainClassifier, DATA_NAME, NUM_LABELS
import wandb

# %%
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
wandb.init()
# %%
data_name = DATA_NAME
model_name = MODEL_CHECKPOINT.split("/")[-1]
data = f"../data/{data_name}"
threshold_high = 0.9995
threshold_low = 0.3
epsilon = 1e-8  # Small constant for loss stabilization
checkpoint_epoch = 1
checkpoint = f"./checkpoint/{model_name}-{data_name}/initial/epoch_{checkpoint_epoch}"

dataset = load_from_disk(data)

# %%
tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT, model_max_length=128)

# %%
def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True)

tokenized_datasets = dataset.map(tokenize_function, load_from_cache_file=True)

# %%
tokenized_datasets.set_format("torch")

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

train_dataset = tokenized_datasets["balanced_train"].remove_columns(['text'])
validation_dataset = tokenized_datasets["balanced_test"].remove_columns(['text'])

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, collate_fn=data_collator)
validation_loader = DataLoader(validation_dataset, batch_size=16, shuffle=False, collate_fn=data_collator)

# %%
model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=NUM_LABELS, output_hidden_states=True)

num_layers = model.config.num_hidden_layers
bias_detectors = [BiasDetector(input_dim=768) for _ in range(num_layers)]
for l, bias_detector in enumerate(bias_detectors):
    bias_detector_checkpoint = f"./checkpoint/{model_name}-{data_name}/initial/bias_detector_layer_{l+1}_epoch_{checkpoint_epoch}.pth"
    bias_detector.load_state_dict(torch.load(bias_detector_checkpoint))

domain_classifiers = [DomainClassifier(input_dim=768) for _ in range(num_layers)]
optimizers_domain = [Adam(domain_classifier.parameters(), lr=1e-3, weight_decay=0.01) for domain_classifier in domain_classifiers]
schedulers_domain = [StepLR(optimizer_domain, step_size=1, gamma=0.9) for optimizer_domain in optimizers_domain]

optimizer_main = Adam(model.parameters(), lr=2e-5)  
optimizers_bias = [Adam(bias_detector.parameters(), lr=1e-3, weight_decay=0.01) for bias_detector in bias_detectors]

scheduler_main = StepLR(optimizer_main, step_size=1, gamma=0.9)
schedulers_bias = [StepLR(optimizer_bias, step_size=1, gamma=0.9) for optimizer_bias in optimizers_bias]

criterion = [torch.nn.CrossEntropyLoss(), torch.nn.BCEWithLogitsLoss()]

# %%
def train_model(model, bias_detectors, domain_classifiers, data_loader, optimizer_main, optimizers_bias, optimizers_domain, scheduler_main, schedulers_bias, schedulers_domain, criterion, device, threshold_high, threshold_low):
    model.to(device)
    for bias_detector in bias_detectors:
        bias_detector.to(device)
    for domain_classifier in domain_classifiers:
        domain_classifier.to(device)
    model.train()
    for bias_detector in bias_detectors:
        bias_detector.train()
    for domain_classifier in domain_classifiers:
        domain_classifier.train()
    
    # Initialize accuracy tracking
    correct_bias_preds = [0] * len(bias_detectors)
    total_bias_samples = [0] * len(bias_detectors)
    correct_domain_preds = [0] * len(domain_classifiers)
    total_domain_samples = [0] * len(domain_classifiers)

    for batch in tqdm(data_loader):
        batch = {k: v.to(device) for k, v in batch.items()}
        
        optimizer_main.zero_grad()
        for optimizer_bias in optimizers_bias:
            optimizer_bias.zero_grad()
        for optimizer_domain in optimizers_domain:
            optimizer_domain.zero_grad()
        
        outputs = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'], output_hidden_states=True)
        intermediate_outputs = [layer[:, 0].detach() for layer in outputs.hidden_states[1:]]  # Skip the embedding layer and get the CLS token

        logits = outputs.logits
        main_loss = criterion[0](logits, batch['labels'])

        bias_loss_total = 0
        domain_loss_total = 0
        for l, (intermediate_output, bias_detector, domain_classifier) in enumerate(zip(intermediate_outputs, bias_detectors, domain_classifiers), start=1):
            bias_preds = bias_detector(intermediate_output).squeeze()  # Process the CLS token from each layer
            predictions = torch.argmax(logits, dim=-1)
            correct = (predictions == batch['labels']).float()
            bias_loss = criterion[1](bias_preds, correct)
            bias_loss_total += bias_loss

            # Update bias detector accuracy tracking
            correct_bias_preds[l-1] += (bias_preds.round() == correct).sum().item()
            total_bias_samples[l-1] += correct.size(0)

            mask_high_bias = bias_preds > threshold_high
            mask_low_bias = bias_preds < threshold_low
            misclassified = (predictions != batch['labels'])

            protected_data_high_bias = intermediate_output[mask_high_bias]
            protected_data_low_bias = intermediate_output[mask_low_bias | misclassified]
            unprotected_data = intermediate_output[~(mask_high_bias | mask_low_bias | misclassified)]

            if protected_data_high_bias.size(0) > 0:
                reverse_protected_high = ReverseLayerF.apply(protected_data_high_bias, 1.0)
                domain_output_protected_high = domain_classifier(reverse_protected_high)
                domain_labels_protected_high = torch.ones(protected_data_high_bias.size(0), device=device, dtype=torch.long)
                domain_loss_protected_high = criterion[0](domain_output_protected_high, domain_labels_protected_high) + epsilon
                # Update domain classifier accuracy tracking
                correct_domain_preds[l-1] += (domain_output_protected_high.argmax(dim=-1) == domain_labels_protected_high).sum().item()
                total_domain_samples[l-1] += domain_labels_protected_high.size(0)
            else:
                domain_loss_protected_high = torch.tensor(0.0, device=device)

            if protected_data_low_bias.size(0) > 0:
                reverse_protected_low = ReverseLayerF.apply(protected_data_low_bias, 1.0)
                domain_output_protected_low = domain_classifier(reverse_protected_low)
                domain_labels_protected_low = torch.ones(protected_data_low_bias.size(0), device=device, dtype=torch.long)
                domain_loss_protected_low = criterion[0](domain_output_protected_low, domain_labels_protected_low) + epsilon
                # Update domain classifier accuracy tracking
                correct_domain_preds[l-1] += (domain_output_protected_low.argmax(dim=-1) == domain_labels_protected_low).sum().item()
                total_domain_samples[l-1] += domain_labels_protected_low.size(0)
            else:
                domain_loss_protected_low = torch.tensor(0.0, device=device)

            if unprotected_data.size(0) > 0:
                reverse_unprotected = ReverseLayerF.apply(unprotected_data, 1.0)
                domain_output_unprotected = domain_classifier(reverse_unprotected)
                domain_labels_unprotected = torch.zeros(unprotected_data.size(0), device=device, dtype=torch.long)
                domain_loss_unprotected = criterion[0](domain_output_unprotected, domain_labels_unprotected) + epsilon
                # Update domain classifier accuracy tracking
                correct_domain_preds[l-1] += (domain_output_unprotected.argmax(dim=-1) == domain_labels_unprotected).sum().item()
                total_domain_samples[l-1] += domain_labels_unprotected.size(0)
            else:
                domain_loss_unprotected = torch.tensor(0.0, device=device)

            domain_loss = domain_loss_protected_high + domain_loss_protected_low + domain_loss_unprotected
            domain_loss_total += domain_loss

        loss = main_loss + bias_loss_total + domain_loss_total
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        for bias_detector in bias_detectors:
            torch.nn.utils.clip_grad_norm_(bias_detector.parameters(), max_norm=1.0)
        for domain_classifier in domain_classifiers:
            torch.nn.utils.clip_grad_norm_(domain_classifier.parameters(), max_norm=1.0)

        optimizer_main.step()
        for optimizer_bias in optimizers_bias:
            optimizer_bias.step()
        for optimizer_domain in optimizers_domain:
            optimizer_domain.step()

        scheduler_main.step()
        for scheduler_bias in schedulers_bias:
            scheduler_bias.step()
        for scheduler_domain in schedulers_domain:
            scheduler_domain.step()

        wandb.log({"loss": loss.item(), "main_loss": main_loss.item(), "bias_loss_total": bias_loss_total.item(), "domain_loss_total": domain_loss_total.item()})

    # Calculate and log accuracy for bias detectors and domain classifiers
    bias_accuracies = []
    domain_accuracies = []
    for l in range(len(bias_detectors)):
        bias_accuracy = correct_bias_preds[l] / total_bias_samples[l] if total_bias_samples[l] > 0 else 0
        domain_accuracy = correct_domain_preds[l] / total_domain_samples[l] if total_domain_samples[l] > 0 else 0
        bias_accuracies.append(bias_accuracy)
        domain_accuracies.append(domain_accuracy)
        wandb.log({f"bias_detector_accuracy_layer_{l+1}": bias_accuracy, f"domain_classifier_accuracy_layer_{l+1}": domain_accuracy})
    
    return bias_accuracies, domain_accuracies

# %%
def validate_model(model, bias_detectors, domain_classifiers, validation_loader, criterion, device, threshold_high, threshold_low):
    model.to(device)
    for bias_detector in bias_detectors:
        bias_detector.to(device)
    for domain_classifier in domain_classifiers:
        domain_classifier.to(device)
    model.eval()
    for bias_detector in bias_detectors:
        bias_detector.eval()
    for domain_classifier in domain_classifiers:
        domain_classifier.eval()

    total_loss, total_correct, total_samples = 0, 0, 0
    correct_bias_preds = [0] * len(bias_detectors)
    total_bias_samples = [0] * len(bias_detectors)
    correct_domain_preds = [0] * len(domain_classifiers)
    total_domain_samples = [0] * len(domain_classifiers)

    with torch.no_grad():
        for batch in tqdm(validation_loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'], output_hidden_states=True)
            intermediate_outputs = [layer[:, 0] for layer in outputs.hidden_states[1:]]  # Get the CLS token from each layer

            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1)
            correct = (predictions == batch['labels']).float()

            main_loss = criterion[0](logits, batch['labels'])

            bias_loss_total = 0
            domain_loss_total = 0
            for l, (intermediate_output, bias_detector, domain_classifier) in enumerate(zip(intermediate_outputs, bias_detectors, domain_classifiers), start=1):
                bias_preds = bias_detector(intermediate_output).squeeze()
                bias_loss = criterion[1](bias_preds, correct)
                bias_loss_total += bias_loss

                # Update bias detector accuracy tracking
                correct_bias_preds[l-1] += ((bias_preds > 0.5).float() == correct).sum().item()
                total_bias_samples[l-1] += correct.size(0)

                domain_loss, correct_domain_preds_layer, total_domain_samples_layer = apply_domain_classification(
                    intermediate_output, bias_preds, predictions, domain_classifier, batch['labels'], device, threshold_high, threshold_low, criterion)
                domain_loss_total += domain_loss

                # Update domain classifier accuracy tracking
                correct_domain_preds[l-1] += correct_domain_preds_layer
                total_domain_samples[l-1] += total_domain_samples_layer

            total_loss += main_loss.item() + bias_loss_total.item() + domain_loss_total.item()
            total_correct += correct.sum().item()
            total_samples += batch['labels'].size(0)

    avg_loss = total_loss / len(validation_loader)
    accuracy = total_correct / total_samples

    # Calculate and log accuracy for bias detectors and domain classifiers
    bias_accuracies = []
    domain_accuracies = []
    for l in range(len(bias_detectors)):
        bias_accuracy = correct_bias_preds[l] / total_bias_samples[l] if total_bias_samples[l] > 0 else 0
        domain_accuracy = correct_domain_preds[l] / total_domain_samples[l] if total_domain_samples[l] > 0 else 0
        bias_accuracies.append(bias_accuracy)
        domain_accuracies.append(domain_accuracy)
        wandb.log({f"val_bias_detector_accuracy_layer_{l+1}": bias_accuracy, f"val_domain_classifier_accuracy_layer_{l+1}": domain_accuracy})

    wandb.log({
        "val_loss": avg_loss, 
        "val_accuracy": accuracy
    })
    print(f"Validation - Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")
    
    return bias_accuracies, domain_accuracies

def apply_domain_classification(intermediate_output, bias_preds, predictions, domain_classifier, labels, device, threshold_high, threshold_low, criterion):
    mask_high_bias = (bias_preds > threshold_high).squeeze()
    mask_low_bias = (bias_preds < threshold_low).squeeze()
    misclassified = (predictions != labels)

    protected_data_high_bias = intermediate_output[mask_high_bias]
    protected_data_low_bias = intermediate_output[mask_low_bias | misclassified]
    unprotected_data = intermediate_output[~(mask_high_bias | mask_low_bias | misclassified)]

    domain_loss = torch.tensor(0.0, device=device)
    correct_domain_preds, total_domain_samples = 0, 0

    if protected_data_high_bias.size(0) > 0:
        reverse_protected_high = ReverseLayerF.apply(protected_data_high_bias, 1.0)
        domain_output_protected_high = domain_classifier(reverse_protected_high)
        domain_labels_protected_high = torch.ones(protected_data_high_bias.size(0), device=device, dtype=torch.long)
        domain_loss += criterion[0](domain_output_protected_high, domain_labels_protected_high)
        correct_domain_preds += (domain_output_protected_high.argmax(dim=-1) == domain_labels_protected_high).sum().item()
        total_domain_samples += domain_labels_protected_high.size(0)

    if protected_data_low_bias.size(0) > 0:
        reverse_protected_low = ReverseLayerF.apply(protected_data_low_bias, 1.0)
        domain_output_protected_low = domain_classifier(reverse_protected_low)
        domain_labels_protected_low = torch.ones(protected_data_low_bias.size(0), device=device, dtype=torch.long)
        domain_loss += criterion[0](domain_output_protected_low, domain_labels_protected_low)
        correct_domain_preds += (domain_output_protected_low.argmax(dim=-1) == domain_labels_protected_low).sum().item()
        total_domain_samples += domain_labels_protected_low.size(0)

    if unprotected_data.size(0) > 0:
        reverse_unprotected = ReverseLayerF.apply(unprotected_data, 1.0)
        domain_output_unprotected = domain_classifier(reverse_unprotected)
        domain_labels_unprotected = torch.zeros(unprotected_data.size(0), device=device, dtype=torch.long)
        domain_loss += criterion[0](domain_output_unprotected, domain_labels_unprotected)
        correct_domain_preds += (domain_output_unprotected.argmax(dim=-1) == domain_labels_unprotected).sum().item()
        total_domain_samples += domain_labels_unprotected.size(0)

    return domain_loss, correct_domain_preds, total_domain_samples


# %%
device = 'cuda' if torch.cuda.is_available() else 'cpu'

output_dir = f"./checkpoint/{model_name}-{data_name}/dann"
if os.path.exists(output_dir):
    # Remove it if it exists
    shutil.rmtree(output_dir)
epochs = 1 # Set the number of epochs

val_bias_accuracies, val_domain_accuracies = validate_model(model, bias_detectors, domain_classifiers, validation_loader, criterion, device, threshold_high, threshold_low)
print(f"Before training Validation Bias Detector Accuracies: {val_bias_accuracies}")

for epoch in range(epochs):
    print(f"Starting Epoch {epoch+1}/{epochs}")
    train_bias_accuracies, train_domain_accuracies = train_model(model, bias_detectors, domain_classifiers, train_loader, optimizer_main, optimizers_bias, optimizers_domain, scheduler_main, schedulers_bias, schedulers_domain, criterion, device, threshold_high, threshold_low)
    val_bias_accuracies, val_domain_accuracies = validate_model(model, bias_detectors, domain_classifiers, validation_loader, criterion, device, threshold_high, threshold_low)
    print(f"Completed Epoch {epoch+1}/{epochs}")

    # Save the model at the end of each epoch
    model_save_path = os.path.join(output_dir, f"epoch_{epoch+1}")
    os.makedirs(model_save_path, exist_ok=True)
    model.config.output_hidden_states = False
    model.save_pretrained(model_save_path)
    for l, bias_detector in enumerate(bias_detectors):
        bias_detector_save_path = os.path.join(output_dir, f"bias_detector_layer_{l+1}_epoch_{epoch+1}.pth")
        torch.save(bias_detector.state_dict(), bias_detector_save_path)

    # Print the accuracies
    print(f"Epoch {epoch+1} - Training Bias Detector Accuracies: {train_bias_accuracies}")
    print(f"Epoch {epoch+1} - Training Domain Classifier Accuracies: {train_domain_accuracies}")
    print(f"Epoch {epoch+1} - Validation Bias Detector Accuracies: {val_bias_accuracies}")
    print(f"Epoch {epoch+1} - Validation Domain Classifier Accuracies: {val_domain_accuracies}")

wandb.finish()
# %%
