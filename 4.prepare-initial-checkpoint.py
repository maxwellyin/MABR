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
from utils.util import MODEL_CHECKPOINT,NUM_LABELS, BiasDetector, DATA_NAME
import wandb
# %%
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
wandb.init()
# %%
data_name = DATA_NAME
model_name = MODEL_CHECKPOINT.split("/")[-1]
data = f"../data/{data_name}"
checkpoint_epoch = 1

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

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=data_collator)
validation_loader = DataLoader(validation_dataset, batch_size=16, shuffle=False, collate_fn=data_collator)

# %%
model = AutoModelForSequenceClassification.from_pretrained(MODEL_CHECKPOINT, num_labels=NUM_LABELS, output_hidden_states=True)

num_layers = model.config.num_hidden_layers
bias_detectors = [BiasDetector(input_dim=768) for _ in range(num_layers)]

optimizer_main = Adam(model.parameters(), lr=2e-5)  
optimizers_bias = [Adam(bias_detector.parameters(), lr=1e-3) for bias_detector in bias_detectors]

criterion = [torch.nn.CrossEntropyLoss(), torch.nn.BCEWithLogitsLoss()]

# %%
def train_initial_model(model, bias_detectors, data_loader, optimizer_main, optimizers_bias, criterion, device):
    model.to(device)
    for bias_detector in bias_detectors:
        bias_detector.to(device)
    model.train()
    for bias_detector in bias_detectors:
        bias_detector.train()
    
    for batch in tqdm(data_loader):
        batch = {k: v.to(device) for k, v in batch.items()}
        
        optimizer_main.zero_grad()
        for optimizer_bias in optimizers_bias:
            optimizer_bias.zero_grad()
        
        outputs = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'], output_hidden_states=True)
        intermediate_outputs = [hidden_state.detach() for hidden_state in outputs.hidden_states]  # Detach all hidden states

        logits = outputs.logits
        main_loss = criterion[0](logits, batch['labels'])

        bias_loss_total = 0
        for l, (intermediate_output, bias_detector) in enumerate(zip(intermediate_outputs[1:], bias_detectors), start=1):  # Skip the input embeddings
            bias_preds = bias_detector(intermediate_output[:, 0])  # Using the CLS token of the current layer
            predictions = torch.argmax(logits, dim=-1)
            correct = (predictions == batch['labels']).float()
            bias_loss = criterion[1](bias_preds.squeeze(), correct)
            bias_loss_total += bias_loss

        loss = main_loss + bias_loss_total
        loss.backward()

        optimizer_main.step()
        for optimizer_bias in optimizers_bias:
            optimizer_bias.step()

        wandb.log({"loss": loss.item(), "main_loss": main_loss.item(), "bias_loss_total": bias_loss_total.item()})

# %%
def validate_model(model, bias_detectors, validation_loader, criterion, device):
    model.to(device)
    for bias_detector in bias_detectors:
        bias_detector.to(device)
    model.eval()
    for bias_detector in bias_detectors:
        bias_detector.eval()

    total_loss, total_correct, total_samples = 0, 0, 0
    total_bias_loss = 0

    with torch.no_grad():
        for batch in tqdm(validation_loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'], output_hidden_states=True)
            intermediate_outputs = [hidden_state.detach() for hidden_state in outputs.hidden_states]  # Detach all hidden states

            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1)
            correct = (predictions == batch['labels']).float()
            
            main_loss = criterion[0](logits, batch['labels'])
            
            bias_loss_total = 0
            for l, (intermediate_output, bias_detector) in enumerate(zip(intermediate_outputs[1:], bias_detectors), start=1):  # Skip the input embeddings
                bias_preds = bias_detector(intermediate_output[:, 0])  # Using the CLS token of the current layer
                bias_loss = criterion[1](bias_preds.squeeze(), correct)
                bias_loss_total += bias_loss

            total_loss += main_loss.item() + bias_loss_total.item()
            total_correct += correct.sum().item()
            total_samples += batch['labels'].size(0)
            total_bias_loss += bias_loss_total.item()

    avg_loss = total_loss / len(validation_loader)
    accuracy = total_correct / total_samples
    avg_bias_loss = total_bias_loss / len(validation_loader)
    
    wandb.log({
        "val_loss": avg_loss, 
        "val_accuracy": accuracy, 
        "val_bias_loss": avg_bias_loss
    })
    print(f"Validation - Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}, Bias Loss: {avg_bias_loss:.4f}")

# %%
device = 'cuda' if torch.cuda.is_available() else 'cpu'

output_dir = f"./checkpoint/{model_name}-{data_name}/initial"
if os.path.exists(output_dir):
    # Remove it if it exists
    shutil.rmtree(output_dir)
epochs = 3  # Train for one epoch
for epoch in range(epochs):
    print(f"Starting Epoch {epoch+1}/{epochs}")
    train_initial_model(model, bias_detectors, train_loader, optimizer_main, optimizers_bias, criterion, device)
    validate_model(model, bias_detectors, validation_loader, criterion, device)
    print(f"Completed Epoch {epoch+1}/{epochs}")

    # Save the model at the end of the epoch
    model_save_path = os.path.join(output_dir, f"epoch_{epoch+1}")
    os.makedirs(model_save_path, exist_ok=True)
    model.config.output_hidden_states = False
    model.save_pretrained(model_save_path)
    for l, bias_detector in enumerate(bias_detectors):
        bias_detector_save_path = os.path.join(output_dir, f"bias_detector_layer_{l+1}_epoch_{epoch+1}.pth")
        torch.save(bias_detector.state_dict(), bias_detector_save_path)

wandb.finish()
# %%
