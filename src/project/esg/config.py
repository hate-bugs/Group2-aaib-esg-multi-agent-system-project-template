from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(slots=True)
class ESGExperimentConfig:
    dataset_path: Path | None = None
    sample_size: int = 10
    chunk_size: int = 180
    chunk_overlap: int = 30
    top_k: int = 4
    worker_batch_size: int = 2
    critique_max_iterations: int = 3
    trials: int = 3
    score_weights: dict[str, float] = field(
        default_factory=lambda: {
            "environmental": 1.0 / 3.0,
            "social": 1.0 / 3.0,
            "governance": 1.0 / 3.0,
        }
    )
    coverage_weights: dict[str, float] = field(
        default_factory=lambda: {
            "environmental": 1.0,
            "social": 1.0,
            "governance": 1.0,
        }
    )
    coverage_threshold: float = 0.2
    output_dir: Path = Path("outputs/experiments")
    chunk_dir: Path = Path("outputs/indexes")
    sample_output_path: Path = Path("src/project/outputs/sample/first10_metrics_summary.json")

    def with_root(self, root: Path) -> "ESGExperimentConfig":
        dataset_candidates = [
            root / "sustainability-reports-2026-05-29.csv",
            root / "data" / "sustainability-reports-2026-05-29.csv",
        ]
        dataset_path = self.dataset_path
        if dataset_path is None:
            for candidate in dataset_candidates:
                if candidate.exists():
                    dataset_path = candidate
                    break
            else:
                dataset_path = dataset_candidates[0]
        elif not dataset_path.is_absolute():
            dataset_path = root / dataset_path

        return ESGExperimentConfig(
            dataset_path=dataset_path,
            sample_size=self.sample_size,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            top_k=self.top_k,
            worker_batch_size=self.worker_batch_size,
            critique_max_iterations=self.critique_max_iterations,
            trials=self.trials,
            score_weights=dict(self.score_weights),
            coverage_weights=dict(self.coverage_weights),
            coverage_threshold=self.coverage_threshold,
            output_dir=root / self.output_dir,
            chunk_dir=root / self.chunk_dir,
            sample_output_path=root / self.sample_output_path,
        )
