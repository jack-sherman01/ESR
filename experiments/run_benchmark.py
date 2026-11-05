"""Entry point: run a single ESRB benchmark instance."""

from __future__ import annotations

import argparse

import hydra
from omegaconf import DictConfig


@hydra.main(config_path="../configs/experiment", config_name="default", version_base=None)
def main(cfg: DictConfig) -> None:
    from esrb.benchmark import GenerationProtocol
    from esrb.benchmark.hazard import HazardModel, HazardType
    from esrb.detection.risk_selection import RiskSensitiveSelector
    # Concrete task / oracle / operator resolved from cfg at runtime
    print(f"[run_benchmark] Starting {cfg.benchmark.name} "
          f"({cfg.benchmark.num_generations} generations)")


if __name__ == "__main__":
    main()
