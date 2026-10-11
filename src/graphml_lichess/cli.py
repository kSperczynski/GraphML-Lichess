import argparse
from pathlib import Path

from graphml_lichess.training.pipeline import TrainingPipeline


def cli():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Training
    parser_train = subparsers.add_parser("train", help="Train the model from scratch")
    parser_train.add_argument("--config", type=Path, required=True, help="Path to the YAML configuration file")

    # parser_tune = subparsers.add_parser("tune", help="Run hyperparameter tuning")

    args = parser.parse_args()
    if args.command == "train":
        print("Starting training pipeline")
        training_pipeline = TrainingPipeline(config_path=args.config)
        training_pipeline.execute()

    elif args.command == "tune":
        raise NotImplementedError("Still need to implement tuning")


if __name__ == "__main__":
    cli()
