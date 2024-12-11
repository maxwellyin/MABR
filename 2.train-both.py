# %%
import os
import torch
from torch.optim import Adam
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.nn.functional import one_hot
from datasets import load_from_disk, DatasetDict, Value
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, DataCollatorWithPadding
from utils.util import debiased_focal_loss, MODEL_CHECKPOINT, NUM_LABELS, DATA_NAME, BiasDetector
import wandb
# %%
wandb.init()
# %%
checkpoint = MODEL_CHECKPOINT
data_name = DATA_NAME
model_name = checkpoint.split("/")[-1]
data = f"../data/{data_name}"

dataset = load_from_disk(data)
# %%
tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT, model_max_length=128)  # 
# %%
def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True)

tokenized_datasets = dataset.map(tokenize_function, load_from_cache_file=True)
# %%
tokenized_datasets.set_format("torch")

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

train_dataset = tokenized_datasets["balanced_train"].remove_columns(['text'])

validation_dataset = tokenized_datasets["balanced_test"].remove_columns(['text'])

train_loader = DataLoader(
    train_dataset, 
    batch_size=16,  
    shuffle=True,  
    collate_fn=data_collator  
)

validation_loader = DataLoader(
    validation_dataset,
    batch_size=16,  
    shuffle=False,  # Usually, we don't shuffle the validation set.
    collate_fn=data_collator  
)

# %%
model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=NUM_LABELS,  output_hidden_states=True)

bias_detector = BiasDetector(input_dim=768)

optimizer_main = Adam(model.parameters(), lr=1e-5)
optimizer_bias = Adam(bias_detector.parameters(), lr=1e-4)
criterion = [torch.nn.CrossEntropyLoss(), torch.nn.BCEWithLogitsLoss()]
# %%
def train_model(model, bias_detector, data_loader, optimizer_main, optimizer_bias, criterion, device):
    model.to(device)
    bias_detector.to(device)
    model.train()
    bias_detector.train()
    
    for batch in tqdm(data_loader):
        batch = {k: v.to(device) for k, v in batch.items()}
        
        optimizer_main.zero_grad()
        optimizer_bias.zero_grad()
        

        outputs = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'], output_hidden_states=True)
        intermediate_output = outputs.hidden_states[-1][:, 0].detach()  # Using last layer's CLS token

        # Determine correctness
        logits = outputs.logits
        main_loss = criterion[0](logits, batch['labels'])

        bias_preds = bias_detector(intermediate_output)
        predictions = torch.argmax(logits, dim=-1)
        correct = (predictions == batch['labels']).float()
        bias_loss = criterion[1](bias_preds.squeeze(), correct)

        main_loss = debiased_focal_loss(logits, batch['labels'], gamma=2.0, bias_preds=bias_preds)

        loss = main_loss + bias_loss
        loss.backward()
        optimizer_main.step()
        optimizer_bias.step()

        wandb.log({"loss": loss.item(), "main_loss": main_loss.item(), "bias_loss": bias_loss.item()})

def validate_model(model, bias_detector, validation_loader, device):
    model.eval()
    bias_detector.eval()
    total_loss, total_correct, total_samples = 0, 0, 0
    with torch.no_grad():
        for batch in tqdm(validation_loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'], output_hidden_states=True)
            intermediate_output = outputs.hidden_states[-1][:, 0]  # Using last layer's CLS token

            # Determine correctness
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1)
            correct = (predictions == batch['labels']).float()
            
            # Forward pass through the bias detector
            bias_preds = bias_detector(intermediate_output)
            loss = criterion[0](logits, batch['labels']) + criterion[1](bias_preds.squeeze(), correct)
            total_loss += loss.item()
            total_correct += (torch.argmax(logits, dim=-1) == batch['labels']).sum().item()
            total_samples += batch['labels'].size(0)
    
    avg_loss = total_loss / len(validation_loader)
    accuracy = total_correct / total_samples
    wandb.log({"val_loss": avg_loss, "val_accuracy": accuracy})
    print(f"Validation - Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")
# %%
device = 'cuda' if torch.cuda.is_available() else 'cpu'
epochs = 3  # Set the number of epochs
for epoch in range(epochs):
    print(f"Starting Epoch {epoch+1}/{epochs}")
    train_model(model, bias_detector, train_loader,  optimizer_main, optimizer_bias,criterion, device)
    validate_model(model, bias_detector, validation_loader, device)
    print(f"Completed Epoch {epoch+1}/{epochs}")

    # Save the model at the end of each epoch
    model_save_path = f"./checkpoint/{model_name}-{data_name}/blind/epoch_{epoch+1}"
    os.makedirs(model_save_path, exist_ok=True)
    model.config.output_hidden_states = False
    model.save_pretrained(model_save_path)

    bias_save_path = f"./checkpoint/{model_name}-{data_name}/bias"
    os.makedirs(bias_save_path, exist_ok=True)
    torch.save(bias_detector.state_dict(), os.path.join(bias_save_path, f"bias_detector_epoch_{epoch+1}.pth"))

wandb.finish()
# %%
