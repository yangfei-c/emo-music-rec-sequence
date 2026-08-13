"""用合成数据检查通用训练引擎、checkpoint、续训和独立评估。"""

import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from msmmr.metrics import compute_va_metrics
from msmmr.training.engine import RunOptions, TaskSpec, prepare_run, run_experiment
from validation.helpers import require


class TinyDataset(Dataset):
    def __init__(self, size: int = 8) -> None:
        generator = torch.Generator().manual_seed(7)
        self.features = torch.randn(size, 3, generator=generator)
        self.targets = torch.tanh(self.features[:, :2])

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, index: int) -> dict:
        return {"features": self.features[index], "targets": self.targets[index]}


def forward(model, batch: dict, device: torch.device):
    return model(batch["features"].to(device)), batch["targets"].to(device)


def metrics(predictions: np.ndarray, targets: np.ndarray, _: dict) -> dict:
    return compute_va_metrics(predictions, targets)


TASK = TaskSpec(
    name="synthetic_va",
    dataset_name="synthetic",
    output_key="unused",
    score_key="va_score",
    metric_fn=metrics,
    metric_fields=(("CCC", "va_score"),),
    prediction_payload=lambda predictions, targets, _metrics, _metadata: {
        "predictions": predictions,
        "targets": targets,
    },
    forward_batch=forward,
)


TRAIN_CONFIG = {
    "training": {
        "max_epochs": 2,
        "early_stopping_patience": 2,
        "early_stopping_min_delta": 0.0,
        "amp": False,
        "gradient_clip_norm": 1.0,
        "max_consecutive_amp_skips": 2,
    },
    "scheduler": {"warmup_ratio": 0.0, "min_learning_rate": 1.0e-6},
}


def build():
    datasets = {split: TinyDataset() for split in ("train", "validation", "test")}
    loaders = {
        split: DataLoader(dataset, batch_size=4, shuffle=split == "train")
        for split, dataset in datasets.items()
    }
    model = torch.nn.Linear(3, 2)
    return datasets, loaders, model, torch.optim.AdamW(model.parameters(), lr=1.0e-3)


def execute(root: Path, options: RunOptions) -> dict:
    datasets, loaders, model, optimizer = build()
    output_dir, checkpoint_path, checkpoint = prepare_run(root, options, seed=42)
    return run_experiment(
        task=TASK,
        options=options,
        output_dir=output_dir,
        checkpoint_path=checkpoint_path,
        checkpoint=checkpoint,
        dataset_config={"synthetic": True},
        train_config=TRAIN_CONFIG,
        run_config={"synthetic": True},
        datasets=datasets,
        loaders=loaders,
        model=model,
        criterion=torch.nn.MSELoss(),
        optimizer=optimizer,
        device=torch.device("cpu"),
    )


def main() -> None:
    with TemporaryDirectory(prefix="msmmr_training_check_") as directory:
        root, run_dir = Path(directory), Path(directory) / "run"
        execute(
            root, RunOptions(target_epochs=1, output_dir=run_dir, evaluate_test=True)
        )
        required = {
            "best.pt",
            "last.pt",
            "history.json",
            "run_config.yaml",
            "completed.json",
            "validation_metrics.json",
            "validation_predictions.npz",
            "test_metrics.json",
            "test_predictions.npz",
        }
        require(
            required <= {path.name for path in run_dir.iterdir()}, "首次训练产物不完整"
        )
        execute(
            root,
            RunOptions(
                target_epochs=2, resume_from=run_dir / "last.pt", evaluate_test=True
            ),
        )
        last = torch.load(run_dir / "last.pt", map_location="cpu", weights_only=False)
        require(
            last["epoch"] == 2 and len(last["history"]) == 2,
            "resume 未从下一 epoch 继续",
        )
        execute(root, RunOptions(evaluate_from=run_dir / "best.pt", evaluate_test=True))
        require(
            (run_dir / "validation_metrics.json").is_file()
            and (run_dir / "test_metrics.json").is_file(),
            "独立评估未落盘",
        )
    print("[PASS] 合成训练、best/last 保存、resume 和 validation/test 独立评估全部通过")


if __name__ == "__main__":
    main()
