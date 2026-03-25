from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path


DEFAULT_MODEL_CHECKPOINT = "roberta-base"
DEFAULT_DATASET_NAME = "biosbias"
DEFAULT_NUM_LABELS = 28


@dataclass(slots=True)
class ExperimentConfig:
    model_checkpoint: str = DEFAULT_MODEL_CHECKPOINT
    dataset_name: str = DEFAULT_DATASET_NAME
    num_labels: int = DEFAULT_NUM_LABELS
    max_length: int = 128
    train_split: str = "balanced_train"
    eval_split: str = "balanced_test"
    fairness_split: str = "test"
    protected_attribute: str = "gender"
    batch_size_train: int = 16
    batch_size_eval: int = 16
    base_batch_size_train: int = 32
    base_learning_rate: float = 2e-5
    main_learning_rate: float = 1e-5
    aux_learning_rate: float = 1e-4
    domain_learning_rate: float = 1e-3
    weight_decay: float = 0.01
    base_epochs: int = 5
    blind_epochs: int = 3
    initial_epochs: int = 3
    multilayer_epochs: int = 5
    gamma: float = 2.0
    threshold_high: float = 0.99
    threshold_low: float = 0.3
    checkpoint_epoch: int = 1
    data_root: Path = field(default_factory=lambda: Path("../data"))
    output_root: Path = field(default_factory=lambda: Path("./checkpoint"))
    device: str | None = None
    use_wandb: bool = False
    wandb_project: str = "mabr"

    @property
    def model_name(self) -> str:
        return self.model_checkpoint.split("/")[-1]

    @property
    def dataset_path(self) -> Path:
        return self.data_root / self.dataset_name

    @property
    def experiment_root(self) -> Path:
        return self.output_root / f"{self.model_name}-{self.dataset_name}"

    def stage_dir(self, stage: str) -> Path:
        return self.experiment_root / stage

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["data_root"] = str(self.data_root)
        payload["output_root"] = str(self.output_root)
        payload["dataset_path"] = str(self.dataset_path)
        payload["experiment_root"] = str(self.experiment_root)
        return payload
