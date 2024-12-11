# %%
import os
import shutil
import torch
from torch.nn.functional import one_hot
from datasets import load_from_disk, DatasetDict, Value
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, DataCollatorWithPadding
from utils.util import compute_metrics, MODEL_CHECKPOINT, NUM_LABELS, DATA_NAME
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
# %%
model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=NUM_LABELS)
# %%
output_dir=f"./checkpoint/{model_name}-{data_name}/base"

# Check if the directory exists
if os.path.exists(output_dir):
    # Remove it if it exists
    shutil.rmtree(output_dir)

# Recreate the empty output directory
os.makedirs(output_dir)

training_args = TrainingArguments(
    output_dir=output_dir,
    logging_strategy="epoch",  # Log training and validation loss at each epoch
    evaluation_strategy="epoch",  # Evaluate at the end of each epoch
    save_strategy="epoch",  # Save the model at the end of each epoch
    learning_rate=2e-5,  
    per_device_train_batch_size=32,  
    per_device_eval_batch_size=16,  
    num_train_epochs=5,      
    load_best_model_at_end=True,  # Optionally load the best model found during training at the end
    metric_for_best_model='accuracy',  # Choose a metric for the best model if load_best_model_at_end is True
    greater_is_better=True  # Used with metric_for_best_model
)

# Note: Hugging Face automatically handles gradient_accumulation_steps and fp16 if enabled via Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets['train'],
    eval_dataset=tokenized_datasets['balanced_test'],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)
# %%
trainer.train()
# %%
trainer.evaluate()
# %%
