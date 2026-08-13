"""中英文文本 VA 实验装配。"""

from pathlib import Path

import torch

from msmmr.data import build_text_dataloader, build_text_datasets
from msmmr.training.utilities import get_device, load_yaml, set_seed
from msmmr.training.factories import (
    build_text_model,
    build_text_optimizer,
    build_va_loss,
)
from msmmr.training.engine import RunOptions, execute_experiment
from msmmr.training.tasks import TEXT_VA_TASK


ROOT = Path(__file__).resolve().parents[2]


def run_text_emotion(options: RunOptions):
    dataset_config = load_yaml(ROOT / "configs/datasets.yaml")
    train_config = load_yaml(ROOT / "configs/text_training.yaml")
    seed = int(train_config.get("seed", 42))
    set_seed(seed)
    device = get_device(str(train_config.get("device", "cpu")))
    datasets = build_text_datasets(dataset_config["text_va"])
    data = train_config["data"]
    use_batch_ccc = float(train_config["loss"].get("ccc_weight", 0.0)) > 0
    generator = torch.Generator().manual_seed(seed)
    loaders = {
        split: build_text_dataloader(
            dataset,
            batch_size=int(data["batch_size"]),
            shuffle=split == "train",
            num_workers=int(data["num_workers"]),
            pin_memory=bool(data["pin_memory"]),
            drop_last=split == "train" and use_batch_ccc,
            generator=generator if split == "train" else None,
        )
        for split, dataset in datasets.items()
    }
    model = build_text_model(train_config, device)
    return execute_experiment(
        output_root=ROOT / "outputs/text/va/xlm_roberta_base",
        options=options,
        seed=seed,
        task=TEXT_VA_TASK,
        dataset_config=dataset_config,
        train_config=train_config,
        run_config={
            "task": "text_va",
            "split": "local_source_stratified_group",
            "target_transform": "[0,1] -> [-1,1] by 2*y-1",
            "dataset": dataset_config,
            "train": train_config,
        },
        datasets=datasets,
        loaders=loaders,
        model=model,
        criterion=build_va_loss(train_config),
        optimizer=build_text_optimizer(model, train_config),
        device=device,
        setup_lines=(
            "split=local fixed split (language/source stratified; not official)",
            "text VA: [0,1] -> [-1,1] by y_norm=2*y-1",
            f"encoder={train_config['model']['pretrained_name']} | "
            "trainable_last_layers="
            f"{train_config['model']['trainable_encoder_layers']}",
            f"loss={train_config['loss']['name']} | "
            "validation selection=equal-language mean CCC",
        ),
        prepare_output=lambda output_dir: model.save_tokenizer(
            output_dir / "tokenizer"
        ),
    )
