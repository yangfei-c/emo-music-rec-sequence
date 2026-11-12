"""Train fresh models from a completed feature batch in this project.

No arguments: compare frozen CLaMP3, Adapters, residual/distillation, and A/P prompt views;
then independently train TextVA and MusicVA on frozen backbone features.
--check-only checks inputs without creating training output.
"""
import argparse
import copy
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import torch
from emomm import SCHEMA, resolve, run_id, save_yaml, require_device
from emomm.data import read_features
from emomm.train import load_yaml, sha256, write_json

REQUIRED = {
    "three_branch": {"pairs_merge", "pairs_merge_views", "pairs_song_describer", "mtg_music", "mtg_tags"},
    "text_va": {"text_emobank"},
    "music_va": {"music_deam", "music_pmemo"},
}


def select_features(data_config, feature_id, required):
    records = resolve(data_config["outputs"]["records"])
    if feature_id == "latest":
        available = []
        for path in records.glob("*/summary.json"):
            summary = json.loads(path.read_text(encoding="utf-8"))
            if summary.get("schema") == SCHEMA and summary.get("status") == "complete" and required <= set(summary.get("files", {})):
                available.append((path.stat().st_mtime_ns, path.parent.name))
        if not available:
            raise FileNotFoundError("没有新项目生成的完整特征。请先运行 scripts/extract_features.py。")
        feature_id = max(available)[1]
    feature_id = run_id(feature_id)
    summary = json.loads((records / feature_id / "summary.json").read_text(encoding="utf-8"))
    if summary.get("status") != "complete" or summary.get("schema") != SCHEMA:
        raise ValueError("特征批次尚未成功完成，或来源不属于新项目")
    if not required <= set(summary["files"]):
        raise ValueError(f"特征缺失: {sorted(required - set(summary['files']))}")
    folder = resolve(data_config["outputs"]["features"]) / feature_id
    hashes = {}
    for name in sorted(summary["files"]):
        entry = summary["files"][name]
        if entry["name"] != f"{name}.npz":
            raise ValueError("特征文件名不符合本项目格式")
        path = folder / entry["name"]
        hashes[name] = sha256(path)
        if hashes[name] != entry["sha256"]:
            raise ValueError(f"特征内容已改变: {path}")
        values = read_features(path)
        if str(values["feature_id"]) != feature_id or str(values["checkpoint_sha256"]) != data_config["clamp3"]["checkpoint_sha256"]:
            raise ValueError(f"特征批次或 CLaMP3 来源不一致: {path}")
    if "pairs_merge_views" in hashes:
        from emomm.features.lyrics import VIEW_METHOD
        views = read_features(folder / "pairs_merge_views.npz")
        if (str(views.get("base_sha256", "")) != hashes.get("pairs_merge")
                or str(views.get("view_method", "")) != VIEW_METHOD
                or not np.all(views["splits"].astype(str) == "train")):
            raise ValueError("歌词片段必须来自同批 MERGE，且只能用于训练")
    return feature_id, folder, hashes


def duplicate_exclusions(folder, names, policy):
    """Exclude lower-priority duplicate records in memory, retaining original splits."""
    if policy not in {"error", "exclude_lower_priority_split"}:
        raise ValueError("duplicate_audio_policy 无效")
    priorities = {"train": 0, "validation": 1, "test": 2}
    groups = {}
    for name in names:
        if name == "pairs_merge_views":
            # Views inherit their source pair's exclusion; they are not new audio records.
            continue
        with np.load(folder / f"{name}.npz", allow_pickle=False) as archive:
            if "audio_content_hashes" not in archive:
                continue
            for identity, split, digest in zip(archive["ids"].astype(str), archive["splits"].astype(str),
                                               archive["audio_content_hashes"].astype(str), strict=True):
                # NPZ strings are np.str_ scalars; reports must use YAML-safe Python strings.
                groups.setdefault(str(digest), []).append((str(name), str(identity), str(split)))
    excluded, details = {}, []
    for entries in groups.values():
        if len({r[2] for r in entries}) < 2:
            continue
        if policy == "error":
            raise ValueError(f"同一音频跨划分: {entries}")
        keep = max((r[2] for r in entries), key=priorities.get)
        removed = [(name, identity, split) for name, identity, split in entries if split != keep]
        for name, identity, _ in removed:
            excluded.setdefault(name, []).append(identity)
        details.append(dict(retained_split=keep, excluded=[dict(dataset=n, id=i, original_split=s) for n, i, s in removed]))
    return {k: sorted(set(v)) for k, v in excluded.items()}, details


def validate_config(config):
    seeds = config["seeds"]
    if not seeds or len(set(seeds)) != len(seeds) or any(isinstance(s, bool) or not isinstance(s, int) or s < 0 or s >= 2**32 for s in seeds):
        raise ValueError("seeds 必须是互不重复的非负整数列表")
    for name in REQUIRED:
        stage = config[name]
        if stage["model"]["embedding_dim"] != 768:
            raise ValueError("本方案固定使用 768 维 CLaMP3 特征")
        cfg = stage["training"]
        for k, v in cfg.items():
            if k.startswith("batch_size") or k in {"max_epochs", "early_stopping_patience"}:
                if isinstance(v, bool) or not isinstance(v, int) or v < 1:
                    raise ValueError(f"{name}.{k} 必须是正整数")
        if cfg["learning_rate"] <= 0 or cfg["weight_decay"] < 0:
            raise ValueError(f"{name} 的优化器参数无效")
    selected = config["three_branch"]
    if (selected["loss"]["semantic_weight"] <= 0
            or selected["loss"]["semantic_distillation_weight"] <= 0
            or selected["loss"]["semantic_distillation_temperature"] <= 0
            or not 0 < selected["model"]["semantic_output_blend"] < 1):
        raise ValueError("残差蒸馏组要求正的语义/蒸馏权重、温度，以及 (0,1) 内的语义混合系数")
    require_device(config["device"])


def execute(config, data_config, stages, selected_id, feature_id, folder, hashes):
    from emomm.train import three_branch, text_va, music_va
    run_root = resolve(config["outputs"]) / selected_id
    if run_root.exists():
        raise FileExistsError(f"训练目录已经存在，请使用新 run_id: {run_root}")
    excluded, duplicate_report = duplicate_exclusions(folder, hashes, config.get("duplicate_audio_policy", "error"))
    if duplicate_report:
        print(f"跨划分重复音频组: {len(duplicate_report)}；排除记录数: {sum(map(len, excluded.values()))}", flush=True)
    effective = copy.deepcopy(config)
    effective.update(run_id=selected_id, feature_id=feature_id, stages=stages,
                     feature_files={name: {"path": str(folder / f"{name}.npz"), "sha256": digest} for name, digest in hashes.items()},
                     comparison_methods=["clamp3", "adapter", "residual_distill", "ap_multiview"] if "three_branch" in stages else [],
                     text_va_input="clamp3_text",
                     duplicate_audio_exclusions=duplicate_report,
                     source_sha256={str(p.relative_to(ROOT)): sha256(p) for p in
                                    [Path(__file__), *sorted((ROOT / "src").rglob("*.py"))]})
    run_root.mkdir(parents=True)
    save_yaml(run_root / "effective_config.yaml", effective)
    selection = dict(schema=SCHEMA, run_id=selected_id, feature_id=feature_id, models={})
    status = dict(schema=SCHEMA, status="running", completed_stages=[])
    write_json(run_root / "summary.json", status)
    try:
        for name in stages:
            stage_config = copy.deepcopy(config[name])
            stage_config.update(run_id=selected_id, feature_id=feature_id, device=config["device"])
            target = run_root / name
            if name == "three_branch":
                from emomm.features.tag_prompts import prepare_prompt_features
                prompt_features = prepare_prompt_features(data_config, feature_id, folder, hashes["mtg_tags"], config["device"])
                effective["ap_prompt_features"] = prompt_features
                save_yaml(run_root / "effective_config.yaml", effective)
                stage_config["supervision"] = dict(semantic_pairs=[str(folder / f"{n}.npz") for n in ("pairs_merge", "pairs_merge_views", "pairs_song_describer")],
                    mtg_factors=str(folder / "mtg_tags.npz"), music_index=str(folder / "mtg_music.npz"),
                    tag_prompt_views=prompt_features["path"], tag_prompt_views_sha256=prompt_features["sha256"])
                pairs, tags, music = three_branch.load_training_data(stage_config, excluded)
                print("对比方案: 冻结 CLaMP3 / Adapter / 残差 Adapter＋语义保持 / 相同方案＋A/P 标签多表达", flush=True)
                print("A/P 对照: residual_distill（原始单提示）vs ap_multiview（每标签4种表达，每步随机1种）；统一原始提示选模", flush=True)
                print("语义配对数: " + ", ".join(f"{split}={int((pairs['splits'] == split).sum())}"
                      for split in ("train", "validation", "test")), flush=True)
                results = three_branch.train_comparison(stage_config, pairs, tags, music, config["seeds"], target)
                del pairs, tags, music
            elif name == "text_va":
                stage_config["data"] = dict(features=str(folder / "text_emobank.npz"))
                values, digest = text_va.prepare_data(stage_config)
                results = [text_va.train_seed(stage_config, values, digest, seed, target / f"seed_{seed}") for seed in config["seeds"]]
            else:
                values = music_va.prepare_data({n: folder / f"music_{n}.npz" for n in ("deam", "pmemo")}, excluded)
                results = [music_va.train_seed(stage_config, values, seed, target / f"seed_{seed}") for seed in config["seeds"]]
            chosen = (min if name == "music_va" else max)(results,
                key=lambda r: (r["best_validation_score"], r["seed"] if name == "music_va" else -r["seed"]))
            checkpoint = target / f"seed_{chosen['seed']}" / "best.pt"
            aggregate = dict(runs=results, selected_seed=chosen["seed"], test_used_for_selection=False)
            if name == "three_branch":
                aggregate["method"] = "residual_distill"
            write_json(target / "aggregate.json", aggregate)
            selection["models"][name] = dict(seed=chosen["seed"], checkpoint=checkpoint.relative_to(run_root).as_posix(),
                sha256=sha256(checkpoint), validation_score=chosen["best_validation_score"],
                criterion={"three_branch": "factorwise_balanced_retrieval_max", "text_va": "mean_ccc_max", "music_va": "source_equal_pinball_min"}[name])
            if name == "three_branch":
                selection["models"][name]["method"] = "residual_distill"
                status["retrieval_comparison"] = "three_branch/comparison.json"
                status["ap_prompt_comparison"] = "three_branch/ap_prompt_comparison.json"
            elif name == "text_va":
                selection["models"][name].update(input_representation="clamp3_text", backbone_checkpoint_sha256=digest)
            save_yaml(run_root / "selection.yaml", selection)
            status["completed_stages"].append(name)
            write_json(run_root / "summary.json", status)
        status["status"] = "complete"
        write_json(run_root / "summary.json", status)
        print(f"训练完成: {run_root}")
    except BaseException as error:
        status.update(status="failed", error=repr(error))
        write_json(run_root / "summary.json", status)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "config/train.yaml")
    parser.add_argument("--data-config", type=Path, default=ROOT / "config/data.yaml")
    parser.add_argument("--feature-id")
    parser.add_argument("--run-id")
    parser.add_argument("--stage", choices=("all", *REQUIRED), default="all")
    parser.add_argument("--seeds", type=int, nargs="+")
    parser.add_argument("--max-epochs", type=int)
    parser.add_argument("--device", choices=("cpu", "cuda"))
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    config, data_config = load_yaml(resolve(args.config)), load_yaml(resolve(args.data_config))
    for key in ("seeds", "device"):
        if getattr(args, key) is not None:
            config[key] = getattr(args, key)
    if args.max_epochs is not None:
        for name in REQUIRED:
            config[name]["training"]["max_epochs"] = args.max_epochs
    validate_config(config)
    torch.set_num_threads(config["num_threads"])
    stages = list(REQUIRED) if args.stage == "all" else [args.stage]
    required = set.union(*(REQUIRED[s] for s in stages))
    feature_id, folder, hashes = select_features(data_config, args.feature_id or config["feature_id"], required)
    selected_id = run_id(args.run_id or config["run_id"])
    print(f"特征批次: {feature_id}\n训练阶段: {', '.join(stages)}\n随机种子: {config['seeds']}")
    if args.check_only:
        excluded, duplicates = duplicate_exclusions(folder, hashes, config.get("duplicate_audio_policy", "error"))
        print(f"跨划分重复音频组: {len(duplicates)}；训练加载排除记录数: {sum(map(len, excluded.values()))}")
        if "three_branch" in stages:
            from emomm.features.tag_prompts import prepare_prompt_features
            info = prepare_prompt_features(data_config, feature_id, folder, hashes["mtg_tags"], config["device"], check_only=True)
            print(f"A/P 标签多表达: {info['labels']}；每标签 {info['training_views_per_label']} 种训练提示；{info['status']}")
            if info["status"] != "cached":
                print(f"正式训练时将自动补提取 {info['new_texts']} 条标签文本；无需重提取音频。")
        print("检查通过；未创建训练产物。")
        return
    execute(config, data_config, stages, selected_id, feature_id, folder, hashes)


if __name__ == "__main__":
    main()
