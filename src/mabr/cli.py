from __future__ import annotations

import argparse
import logging
from pathlib import Path

from .config import ExperimentConfig


def configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="MABR experiment CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    for command in ["train-base", "train-blind", "prepare-initial", "train-multilayer", "analyze-accuracy", "eval-fairness"]:
        subparser = subparsers.add_parser(command)
        add_common_arguments(subparser)
        if command in {"train-multilayer", "analyze-accuracy"}:
            subparser.add_argument("--checkpoint-epoch", type=int, default=1)
            subparser.add_argument("--threshold-high", type=float, default=0.99)
            subparser.add_argument("--threshold-low", type=float, default=0.3)
        if command == "eval-fairness":
            subparser.add_argument("--protected-attribute", default="gender")
            subparser.add_argument("--fairness-split", default="test")
    return parser


def add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model-checkpoint", default="roberta-base")
    parser.add_argument("--dataset-name", default="biosbias")
    parser.add_argument("--num-labels", type=int, default=28)
    parser.add_argument("--data-root", type=Path, default=Path("../data"))
    parser.add_argument("--output-root", type=Path, default=Path("./checkpoint"))
    parser.add_argument("--device", default=None)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", default="mabr")


def args_to_config(args: argparse.Namespace) -> ExperimentConfig:
    return ExperimentConfig(
        model_checkpoint=args.model_checkpoint,
        dataset_name=args.dataset_name,
        num_labels=args.num_labels,
        data_root=args.data_root,
        output_root=args.output_root,
        device=args.device,
        use_wandb=args.wandb,
        wandb_project=args.wandb_project,
        checkpoint_epoch=getattr(args, "checkpoint_epoch", 1),
        threshold_high=getattr(args, "threshold_high", 0.99),
        threshold_low=getattr(args, "threshold_low", 0.3),
        protected_attribute=getattr(args, "protected_attribute", "gender"),
        fairness_split=getattr(args, "fairness_split", "test"),
    )


def main(argv: list[str] | None = None) -> None:
    configure_logging()
    parser = build_parser()
    args = parser.parse_args(argv)
    config = args_to_config(args)

    if args.command == "train-base":
        from .pipeline import run_base_training

        run_base_training(config)
    elif args.command == "train-blind":
        from .pipeline import run_blind_training

        run_blind_training(config)
    elif args.command == "prepare-initial":
        from .pipeline import run_initial_checkpoint_preparation

        run_initial_checkpoint_preparation(config)
    elif args.command == "train-multilayer":
        from .pipeline import run_multilayer_training

        run_multilayer_training(config, report_layer_accuracy=False)
    elif args.command == "analyze-accuracy":
        from .pipeline import run_multilayer_training

        run_multilayer_training(config, report_layer_accuracy=True)
    elif args.command == "eval-fairness":
        from .pipeline import run_fairness_evaluation

        results = run_fairness_evaluation(config)
        for key, value in results.items():
            print(f"{key}: {value:.6f}")
    else:
        raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
