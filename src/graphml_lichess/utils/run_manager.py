import json
import platform
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch

from graphml_lichess.utils.config import save_yaml


class RunManager:
    def __init__(self, base_dir: Path, prefix: str):
        self.base_dir = Path(base_dir)
        self.prefix = prefix
        self.run_id, self.run_dir = self._create_run_directory()

    def _create_run_directory(self) -> tuple[str, Path]:
        """Creates the directory to save the data of the current run"""
        timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
        run_id = f"{self.prefix}_{timestamp}"
        run_dir = self.base_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=False)
        return run_id, run_dir

    def _write_json(self, data: dict[str, Any], filename: str) -> None:
        path = self.run_dir / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(data, indent=4, default=str),
            encoding="utf-8",
        )

    def save_checkpoint(
        self,
        model: torch.nn.Module,
        model_config: dict[str, Any],
        feature_columns: list[str],
        target_column: str,
        edge_feature_columns: list[str],
        metadata: dict[str, Any],
        filename: str = "checkpoint.pt",
    ) -> None:
        """Saves model state and schema to the run directory."""
        path = self.run_dir / filename
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "model_config": model_config,
                "feature_columns": feature_columns,
                "target_column": target_column,
                "edge_feature_columns": edge_feature_columns,
                "metadata": metadata,
            },
            path,
        )

    def save_split(self, split: dict[str, list[int]], filename: str = "split.json") -> None:
        """Saves data split indices."""
        self._write_json(split, filename)

    def save_config(self, config: dict[str, Any], filename: str = "config.yaml") -> None:
        """Saves the experiment YAML configuration."""
        path = self.run_dir / filename
        save_yaml(config, path)

    @staticmethod
    def collect_metadata(device: torch.device, seed: int) -> dict[str, Any]:
        """Collects environment and git metadata. Kept static as it doesn't rely on run state."""
        try:
            git_commit = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True, timeout=2.0
            ).strip()
        except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
            git_commit = "unknown"

        return {
            "seed": seed,
            "device": str(device),
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "torch_version": torch.__version__,
            "git_commit": git_commit,
        }
