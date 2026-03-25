from __future__ import annotations

import logging
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
from transformers import (
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)

from .config import ExperimentConfig
from .data import (
    build_collator,
    build_dataloader,
    build_tokenizer,
    load_dataset,
    maybe_remove_columns,
    resolve_device,
    tokenize_dataset,
)
from .losses import debiased_focal_loss, focal_loss
from .metrics import (
    calculate_gap,
    calculate_rms_tpr_gap,
    calculate_tpr_fpr,
    compute_accuracy_metrics,
    independence,
    separation,
    sufficiency,
)
from .models import BiasDetector, DomainClassifier, ReverseLayerF

LOGGER = logging.getLogger(__name__)


def maybe_init_wandb(config: ExperimentConfig, run_name: str):
    if not config.use_wandb:
        return None
    try:
        import wandb  # type: ignore
    except ImportError as exc:
        raise RuntimeError("wandb is not installed. Install `mabr[tracking]` or disable --wandb.") from exc
    return wandb.init(project=config.wandb_project, name=run_name, config=config.to_dict())


def maybe_log_wandb(config: ExperimentConfig, payload: dict[str, float]) -> None:
    if not config.use_wandb:
        return
    import wandb  # type: ignore

    wandb.log(payload)


def maybe_finish_wandb(config: ExperimentConfig) -> None:
    if not config.use_wandb:
        return
    import wandb  # type: ignore

    wandb.finish()


def reset_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def get_hidden_size(model: AutoModelForSequenceClassification) -> int:
    hidden_size = getattr(model.config, "hidden_size", None)
    if hidden_size is None:
        raise ValueError("Model config does not expose hidden_size.")
    return int(hidden_size)


def load_model(config: ExperimentConfig, checkpoint: str | Path | None = None, output_hidden_states: bool = False):
    source = str(checkpoint or config.model_checkpoint)
    return AutoModelForSequenceClassification.from_pretrained(
        source,
        num_labels=config.num_labels,
        output_hidden_states=output_hidden_states,
    )


def save_model(model, path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    original_flag = getattr(model.config, "output_hidden_states", False)
    model.config.output_hidden_states = False
    model.save_pretrained(str(path))
    model.config.output_hidden_states = original_flag


def save_bias_detectors(bias_detectors: list[BiasDetector], directory: Path, epoch: int) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for index, bias_detector in enumerate(bias_detectors, start=1):
        torch.save(bias_detector.state_dict(), directory / f"bias_detector_layer_{index}_epoch_{epoch}.pth")


def load_bias_detectors(config: ExperimentConfig, model, stage: str, epoch: int) -> list[BiasDetector]:
    hidden_size = get_hidden_size(model)
    detectors = [BiasDetector(input_dim=hidden_size) for _ in range(model.config.num_hidden_layers)]
    for index, detector in enumerate(detectors, start=1):
        checkpoint = config.stage_dir(stage) / f"bias_detector_layer_{index}_epoch_{epoch}.pth"
        detector.load_state_dict(torch.load(checkpoint, map_location="cpu"))
    return detectors


def build_common_data(config: ExperimentConfig, remove_text: bool = False):
    dataset = load_dataset(config)
    tokenizer = build_tokenizer(config)
    tokenized = tokenize_dataset(dataset, tokenizer, remove_text=remove_text)
    collator = build_collator(tokenizer)
    return dataset, tokenizer, tokenized, collator


def run_base_training(config: ExperimentConfig) -> None:
    maybe_init_wandb(config, "train-base")
    _, tokenizer, tokenized, collator = build_common_data(config)
    model = load_model(config)
    output_dir = config.stage_dir("base")
    reset_dir(output_dir)
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        logging_strategy="epoch",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=config.base_learning_rate,
        per_device_train_batch_size=config.base_batch_size_train,
        per_device_eval_batch_size=config.batch_size_eval,
        num_train_epochs=config.base_epochs,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        report_to=["wandb"] if config.use_wandb else [],
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized[config.eval_split],
        data_collator=collator,
        tokenizer=tokenizer,
        compute_metrics=compute_accuracy_metrics,
    )
    trainer.train()
    trainer.evaluate()
    maybe_finish_wandb(config)


def run_blind_training(config: ExperimentConfig) -> None:
    maybe_init_wandb(config, "train-blind")
    _, _, tokenized, collator = build_common_data(config)
    train_dataset = maybe_remove_columns(tokenized[config.train_split], ["text"])
    validation_dataset = maybe_remove_columns(tokenized[config.eval_split], ["text"])
    train_loader = build_dataloader(train_dataset, collator, config.batch_size_train, shuffle=True)
    validation_loader = build_dataloader(validation_dataset, collator, config.batch_size_eval, shuffle=False)
    device = resolve_device(config.device)

    model = load_model(config, output_hidden_states=True).to(device)
    bias_detector = BiasDetector(input_dim=get_hidden_size(model)).to(device)
    optimizer_main = Adam(model.parameters(), lr=config.main_learning_rate)
    optimizer_bias = Adam(bias_detector.parameters(), lr=config.aux_learning_rate)
    ce_loss = torch.nn.CrossEntropyLoss()
    bce_logits = torch.nn.BCEWithLogitsLoss()

    output_dir = config.stage_dir("blind")
    output_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, config.blind_epochs + 1):
        model.train()
        bias_detector.train()
        for batch in tqdm(train_loader, desc=f"blind epoch {epoch}"):
            batch = {key: value.to(device) for key, value in batch.items()}
            optimizer_main.zero_grad()
            optimizer_bias.zero_grad()

            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                output_hidden_states=True,
            )
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1)
            correct = (predictions == batch["labels"]).float()

            detached_features = outputs.hidden_states[-1][:, 0].detach()
            bias_logits = bias_detector(detached_features)
            bias_probs = torch.sigmoid(bias_logits)
            main_loss = debiased_focal_loss(logits, batch["labels"], config.gamma, bias_probs)
            bias_loss = bce_logits(bias_logits.squeeze(-1), correct)
            loss = main_loss + bias_loss
            loss.backward()
            optimizer_main.step()
            optimizer_bias.step()

            maybe_log_wandb(
                config,
                {
                    "loss": float(loss.item()),
                    "main_loss": float(main_loss.item()),
                    "bias_loss": float(bias_loss.item()),
                },
            )

        metrics = evaluate_blind_model(model, bias_detector, validation_loader, device, ce_loss, bce_logits, config)
        LOGGER.info("Blind epoch %s validation: %s", epoch, metrics)
        save_model(model, output_dir / f"epoch_{epoch}")
        bias_dir = config.experiment_root / "bias"
        bias_dir.mkdir(parents=True, exist_ok=True)
        torch.save(bias_detector.state_dict(), bias_dir / f"bias_detector_epoch_{epoch}.pth")

    maybe_finish_wandb(config)


def evaluate_blind_model(model, bias_detector, dataloader, device, ce_loss, bce_logits, config: ExperimentConfig):
    model.eval()
    bias_detector.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    with torch.no_grad():
        for batch in dataloader:
            batch = {key: value.to(device) for key, value in batch.items()}
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                output_hidden_states=True,
            )
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1)
            correct = (predictions == batch["labels"]).float()
            bias_logits = bias_detector(outputs.hidden_states[-1][:, 0])
            loss = ce_loss(logits, batch["labels"]) + bce_logits(bias_logits.squeeze(-1), correct)
            total_loss += float(loss.item())
            total_correct += int((predictions == batch["labels"]).sum().item())
            total_samples += int(batch["labels"].size(0))
    metrics = {
        "val_loss": total_loss / max(len(dataloader), 1),
        "val_accuracy": total_correct / max(total_samples, 1),
    }
    maybe_log_wandb(config, metrics)
    return metrics


def run_initial_checkpoint_preparation(config: ExperimentConfig) -> None:
    maybe_init_wandb(config, "prepare-initial")
    _, _, tokenized, collator = build_common_data(config)
    train_dataset = maybe_remove_columns(tokenized[config.train_split], ["text"])
    validation_dataset = maybe_remove_columns(tokenized[config.eval_split], ["text"])
    train_loader = build_dataloader(train_dataset, collator, config.batch_size_train, shuffle=True)
    validation_loader = build_dataloader(validation_dataset, collator, config.batch_size_eval, shuffle=False)
    device = resolve_device(config.device)

    model = load_model(config, output_hidden_states=True).to(device)
    hidden_size = get_hidden_size(model)
    bias_detectors = [BiasDetector(input_dim=hidden_size).to(device) for _ in range(model.config.num_hidden_layers)]
    optimizer_main = Adam(model.parameters(), lr=config.base_learning_rate)
    optimizers_bias = [Adam(detector.parameters(), lr=config.domain_learning_rate) for detector in bias_detectors]
    ce_loss = torch.nn.CrossEntropyLoss()
    bce_logits = torch.nn.BCEWithLogitsLoss()

    output_dir = config.stage_dir("initial")
    reset_dir(output_dir)

    for epoch in range(1, config.initial_epochs + 1):
        train_initial_epoch(model, bias_detectors, train_loader, optimizer_main, optimizers_bias, ce_loss, bce_logits, device, config)
        metrics = validate_initial_epoch(model, bias_detectors, validation_loader, ce_loss, bce_logits, device, config)
        LOGGER.info("Initial epoch %s validation: %s", epoch, metrics)
        save_model(model, output_dir / f"epoch_{epoch}")
        save_bias_detectors(bias_detectors, output_dir, epoch)

    maybe_finish_wandb(config)


def train_initial_epoch(model, bias_detectors, dataloader, optimizer_main, optimizers_bias, ce_loss, bce_logits, device, config: ExperimentConfig):
    model.train()
    for detector in bias_detectors:
        detector.train()
    for batch in tqdm(dataloader, desc="initial train"):
        batch = {key: value.to(device) for key, value in batch.items()}
        optimizer_main.zero_grad()
        for optimizer_bias in optimizers_bias:
            optimizer_bias.zero_grad()

        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            output_hidden_states=True,
        )
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)
        correct = (predictions == batch["labels"]).float()
        main_loss = ce_loss(logits, batch["labels"])
        bias_loss_total = torch.tensor(0.0, device=device)
        for hidden_state, detector in zip(outputs.hidden_states[1:], bias_detectors):
            bias_logits = detector(hidden_state[:, 0].detach())
            bias_loss_total = bias_loss_total + bce_logits(bias_logits.squeeze(-1), correct)
        loss = main_loss + bias_loss_total
        loss.backward()
        optimizer_main.step()
        for optimizer_bias in optimizers_bias:
            optimizer_bias.step()
        maybe_log_wandb(
            config,
            {
                "loss": float(loss.item()),
                "main_loss": float(main_loss.item()),
                "bias_loss_total": float(bias_loss_total.item()),
            },
        )


def validate_initial_epoch(model, bias_detectors, dataloader, ce_loss, bce_logits, device, config: ExperimentConfig):
    model.eval()
    for detector in bias_detectors:
        detector.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    total_bias_loss = 0.0
    with torch.no_grad():
        for batch in dataloader:
            batch = {key: value.to(device) for key, value in batch.items()}
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                output_hidden_states=True,
            )
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1)
            correct = (predictions == batch["labels"]).float()
            main_loss = ce_loss(logits, batch["labels"])
            bias_loss_total = torch.tensor(0.0, device=device)
            for hidden_state, detector in zip(outputs.hidden_states[1:], bias_detectors):
                bias_logits = detector(hidden_state[:, 0])
                bias_loss_total = bias_loss_total + bce_logits(bias_logits.squeeze(-1), correct)
            total_loss += float((main_loss + bias_loss_total).item())
            total_correct += int(correct.sum().item())
            total_samples += int(batch["labels"].size(0))
            total_bias_loss += float(bias_loss_total.item())
    metrics = {
        "val_loss": total_loss / max(len(dataloader), 1),
        "val_accuracy": total_correct / max(total_samples, 1),
        "val_bias_loss": total_bias_loss / max(len(dataloader), 1),
    }
    maybe_log_wandb(config, metrics)
    return metrics


def apply_domain_classification(
    intermediate_output: torch.Tensor,
    bias_probs: torch.Tensor,
    predictions: torch.Tensor,
    domain_classifier: DomainClassifier,
    labels: torch.Tensor,
    device: str,
    threshold_high: float,
    threshold_low: float,
    ce_loss,
):
    mask_high_bias = bias_probs > threshold_high
    mask_low_bias = bias_probs < threshold_low
    misclassified = predictions != labels

    protected_data_high_bias = intermediate_output[mask_high_bias]
    protected_data_low_bias = intermediate_output[mask_low_bias | misclassified]
    unprotected_data = intermediate_output[~(mask_high_bias | mask_low_bias | misclassified)]

    domain_loss = torch.tensor(0.0, device=device)
    correct_domain_preds = 0
    total_domain_samples = 0

    if protected_data_high_bias.size(0) > 0:
        reversed_features = ReverseLayerF.apply(protected_data_high_bias, 1.0)
        domain_output = domain_classifier(reversed_features)
        domain_labels = torch.ones(protected_data_high_bias.size(0), device=device, dtype=torch.long)
        domain_loss = domain_loss + ce_loss(domain_output, domain_labels)
        correct_domain_preds += int((domain_output.argmax(dim=-1) == domain_labels).sum().item())
        total_domain_samples += int(domain_labels.size(0))

    if protected_data_low_bias.size(0) > 0:
        reversed_features = ReverseLayerF.apply(protected_data_low_bias, 1.0)
        domain_output = domain_classifier(reversed_features)
        domain_labels = torch.ones(protected_data_low_bias.size(0), device=device, dtype=torch.long)
        domain_loss = domain_loss + ce_loss(domain_output, domain_labels)
        correct_domain_preds += int((domain_output.argmax(dim=-1) == domain_labels).sum().item())
        total_domain_samples += int(domain_labels.size(0))

    if unprotected_data.size(0) > 0:
        reversed_features = ReverseLayerF.apply(unprotected_data, 1.0)
        domain_output = domain_classifier(reversed_features)
        domain_labels = torch.zeros(unprotected_data.size(0), device=device, dtype=torch.long)
        domain_loss = domain_loss + ce_loss(domain_output, domain_labels)
        correct_domain_preds += int((domain_output.argmax(dim=-1) == domain_labels).sum().item())
        total_domain_samples += int(domain_labels.size(0))

    return domain_loss, correct_domain_preds, total_domain_samples


def run_multilayer_training(config: ExperimentConfig, report_layer_accuracy: bool = False) -> None:
    maybe_init_wandb(config, "train-multilayer")
    _, _, tokenized, collator = build_common_data(config)
    train_dataset = maybe_remove_columns(tokenized[config.train_split], ["text"])
    validation_dataset = maybe_remove_columns(tokenized[config.eval_split], ["text"])
    train_loader = build_dataloader(train_dataset, collator, 64, shuffle=True)
    validation_loader = build_dataloader(validation_dataset, collator, config.batch_size_eval, shuffle=False)
    device = resolve_device(config.device)

    checkpoint = config.stage_dir("initial") / f"epoch_{config.checkpoint_epoch}"
    model = load_model(config, checkpoint=checkpoint, output_hidden_states=True).to(device)
    hidden_size = get_hidden_size(model)
    bias_detectors = load_bias_detectors(config, model, "initial", config.checkpoint_epoch)
    bias_detectors = [detector.to(device) for detector in bias_detectors]
    domain_classifiers = [DomainClassifier(input_dim=hidden_size).to(device) for _ in range(model.config.num_hidden_layers)]

    optimizer_main = Adam(model.parameters(), lr=config.base_learning_rate)
    optimizers_bias = [
        Adam(detector.parameters(), lr=config.domain_learning_rate, weight_decay=config.weight_decay)
        for detector in bias_detectors
    ]
    optimizers_domain = [
        Adam(classifier.parameters(), lr=config.domain_learning_rate, weight_decay=config.weight_decay)
        for classifier in domain_classifiers
    ]
    scheduler_main = StepLR(optimizer_main, step_size=1, gamma=0.9)
    schedulers_bias = [StepLR(optimizer, step_size=1, gamma=0.9) for optimizer in optimizers_bias]
    schedulers_domain = [StepLR(optimizer, step_size=1, gamma=0.9) for optimizer in optimizers_domain]
    ce_loss = torch.nn.CrossEntropyLoss()
    bce_logits = torch.nn.BCEWithLogitsLoss()

    output_dir = config.stage_dir("dann")
    reset_dir(output_dir)

    for epoch in range(1, config.multilayer_epochs + 1):
        train_metrics = run_multilayer_epoch(
            model,
            bias_detectors,
            domain_classifiers,
            train_loader,
            optimizer_main,
            optimizers_bias,
            optimizers_domain,
            scheduler_main,
            schedulers_bias,
            schedulers_domain,
            ce_loss,
            bce_logits,
            device,
            config,
            report_layer_accuracy,
            train=True,
        )
        val_metrics = run_multilayer_epoch(
            model,
            bias_detectors,
            domain_classifiers,
            validation_loader,
            optimizer_main,
            optimizers_bias,
            optimizers_domain,
            scheduler_main,
            schedulers_bias,
            schedulers_domain,
            ce_loss,
            bce_logits,
            device,
            config,
            report_layer_accuracy,
            train=False,
        )
        LOGGER.info("Multilayer epoch %s train=%s val=%s", epoch, train_metrics, val_metrics)
        save_model(model, output_dir / f"epoch_{epoch}")
        save_bias_detectors(bias_detectors, output_dir, epoch)

    maybe_finish_wandb(config)


def run_multilayer_epoch(
    model,
    bias_detectors,
    domain_classifiers,
    dataloader,
    optimizer_main,
    optimizers_bias,
    optimizers_domain,
    scheduler_main,
    schedulers_bias,
    schedulers_domain,
    ce_loss,
    bce_logits,
    device,
    config: ExperimentConfig,
    report_layer_accuracy: bool,
    train: bool,
):
    model.train(mode=train)
    for module in [*bias_detectors, *domain_classifiers]:
        module.train(mode=train)

    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    correct_bias_preds = [0] * len(bias_detectors)
    total_bias_samples = [0] * len(bias_detectors)
    correct_domain_preds = [0] * len(domain_classifiers)
    total_domain_samples = [0] * len(domain_classifiers)

    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for batch in tqdm(dataloader, desc="multilayer train" if train else "multilayer eval"):
            batch = {key: value.to(device) for key, value in batch.items()}
            if train:
                optimizer_main.zero_grad()
                for optimizer_bias in optimizers_bias:
                    optimizer_bias.zero_grad()
                for optimizer_domain in optimizers_domain:
                    optimizer_domain.zero_grad()

            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                output_hidden_states=True,
            )
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1)
            correct = (predictions == batch["labels"]).float()
            main_loss = ce_loss(logits, batch["labels"])
            bias_loss_total = torch.tensor(0.0, device=device)
            domain_loss_total = torch.tensor(0.0, device=device)

            for index, (intermediate_output, bias_detector, domain_classifier) in enumerate(
                zip([layer[:, 0] for layer in outputs.hidden_states[1:]], bias_detectors, domain_classifiers)
            ):
                bias_logits = bias_detector(intermediate_output.detach()).squeeze(-1)
                bias_probs = torch.sigmoid(bias_logits)
                bias_loss_total = bias_loss_total + bce_logits(bias_logits, correct)
                correct_bias_preds[index] += int((((bias_probs > 0.5).float()) == correct).sum().item())
                total_bias_samples[index] += int(correct.size(0))

                domain_loss, correct_preds, total_preds = apply_domain_classification(
                    intermediate_output,
                    bias_probs,
                    predictions,
                    domain_classifier,
                    batch["labels"],
                    device,
                    config.threshold_high,
                    config.threshold_low,
                    ce_loss,
                )
                domain_loss_total = domain_loss_total + domain_loss
                correct_domain_preds[index] += correct_preds
                total_domain_samples[index] += total_preds

            loss = main_loss + bias_loss_total + domain_loss_total
            if train:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                for module in [*bias_detectors, *domain_classifiers]:
                    torch.nn.utils.clip_grad_norm_(module.parameters(), max_norm=1.0)
                optimizer_main.step()
                for optimizer in optimizers_bias:
                    optimizer.step()
                for optimizer in optimizers_domain:
                    optimizer.step()
                scheduler_main.step()
                for scheduler in [*schedulers_bias, *schedulers_domain]:
                    scheduler.step()

            total_loss += float(loss.item())
            total_correct += int(correct.sum().item())
            total_samples += int(batch["labels"].size(0))
            maybe_log_wandb(
                config,
                {
                    f"{'train' if train else 'val'}_loss": float(loss.item()),
                    f"{'train' if train else 'val'}_main_loss": float(main_loss.item()),
                    f"{'train' if train else 'val'}_bias_loss_total": float(bias_loss_total.item()),
                    f"{'train' if train else 'val'}_domain_loss_total": float(domain_loss_total.item()),
                },
            )

    metrics = {
        "loss": total_loss / max(len(dataloader), 1),
        "accuracy": total_correct / max(total_samples, 1),
    }
    if report_layer_accuracy:
        for index in range(len(bias_detectors)):
            metrics[f"bias_detector_accuracy_layer_{index + 1}"] = correct_bias_preds[index] / max(total_bias_samples[index], 1)
            metrics[f"domain_classifier_accuracy_layer_{index + 1}"] = correct_domain_preds[index] / max(total_domain_samples[index], 1)
    maybe_log_wandb(config, {f"{'train' if train else 'val'}_{key}": value for key, value in metrics.items()})
    return metrics


def run_fairness_evaluation(config: ExperimentConfig) -> dict[str, float]:
    maybe_init_wandb(config, "fairness-eval")
    raw_dataset, _, tokenized, collator = build_common_data(config, remove_text=True)
    test_dataset = tokenized[config.fairness_split]
    validation_loader = build_dataloader(test_dataset, collator, config.batch_size_eval, shuffle=False)
    checkpoint_dir = config.stage_dir("dann")
    device = resolve_device(config.device)

    results: dict[str, float] = {}
    for checkpoint_path in sorted(checkpoint_dir.iterdir()):
        if not checkpoint_path.is_dir():
            continue
        if not checkpoint_path.name.startswith(("checkpoint", "epoch")):
            continue
        model = load_model(config, checkpoint=checkpoint_path).to(device)
        labels, predictions = evaluate_classifier(model, validation_loader, device, config.gamma)
        protected_values = np.asarray(raw_dataset[config.fairness_split][config.protected_attribute])
        df = pd.DataFrame(
            {
                "label": labels,
                "prediction": predictions,
                config.protected_attribute: protected_values,
            }
        )
        total_accuracy = float((df["label"] == df["prediction"]).mean())
        metrics = calculate_tpr_fpr(df, config.protected_attribute, "label", "prediction", config.num_labels)
        tpr_gap, fpr_gap = calculate_gap(metrics)
        rms_tpr_gap = calculate_rms_tpr_gap(df, "label", "prediction", config.protected_attribute, config.num_labels)
        independence_score = independence(df["prediction"].to_numpy(), df[config.protected_attribute].to_numpy(), config.num_labels)
        separation_score = separation(
            df["prediction"].to_numpy(),
            df["label"].to_numpy(),
            df[config.protected_attribute].to_numpy(),
            config.num_labels,
        )
        sufficiency_score = sufficiency(
            df["prediction"].to_numpy(),
            df["label"].to_numpy(),
            df[config.protected_attribute].to_numpy(),
            config.num_labels,
        )
        prefix = checkpoint_path.name
        results.update(
            {
                f"{prefix}.accuracy": total_accuracy,
                f"{prefix}.tpr_gap": tpr_gap,
                f"{prefix}.fpr_gap": fpr_gap,
                f"{prefix}.rms_tpr_gap": rms_tpr_gap,
                f"{prefix}.independence": independence_score,
                f"{prefix}.separation": separation_score,
                f"{prefix}.sufficiency": sufficiency_score,
            }
        )
        maybe_log_wandb(
            config,
            {
                "total_accuracy": total_accuracy,
                "tpr_gap": tpr_gap,
                "fpr_gap": fpr_gap,
                "rms_tpr_gap": rms_tpr_gap,
                "independence_score": independence_score,
                "separation_score": separation_score,
                "sufficiency_score": sufficiency_score,
            },
        )
    maybe_finish_wandb(config)
    return results


def evaluate_classifier(model, dataloader, device, gamma: float):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    all_labels: list[np.ndarray] = []
    all_predictions: list[np.ndarray] = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="fairness eval"):
            batch = {key: value.to(device) for key, value in batch.items()}
            outputs = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1)
            total_loss += float(focal_loss(logits, batch["labels"], gamma).item())
            total_correct += int((predictions == batch["labels"]).sum().item())
            total_samples += int(batch["labels"].size(0))
            all_labels.append(batch["labels"].cpu().numpy())
            all_predictions.append(predictions.cpu().numpy())
    LOGGER.info(
        "Evaluation loss=%.4f accuracy=%.4f",
        total_loss / max(len(dataloader), 1),
        total_correct / max(total_samples, 1),
    )
    return np.concatenate(all_labels), np.concatenate(all_predictions)
