"""Run in msmmr: raw datasets -> freshly computed MERT/CLaMP3 features.

No arguments: extract all configured datasets into a new feature_id.
--check-only: check source files and models without writing products.
--resume --feature-id ID: continue this project's interrupted extraction.
"""
import argparse
import gc
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import torch
from emomm import SCHEMA, fingerprint, resolve, run_id, save_npz, save_yaml, require_device
from emomm.data import DATASETS, MTGTrack, load_raw, metadata_arrays, select_supported_labels, tag_prompt
from emomm.train import load_yaml, sha256, write_json


def check_pretrained(config):
    from huggingface_hub import snapshot_download
    path = resolve(config["clamp3"]["checkpoint"])
    if sha256(path) != config["clamp3"]["checkpoint_sha256"]:
        raise ValueError("CLaMP3 预训练权重校验失败")
    for model, revision in ((config["mert"]["model"], config["mert"]["revision"]),
                            (config["clamp3"]["text_model"], None)):
        snapshot_download(repo_id=model, revision=revision,
                          local_files_only=config["local_files_only"],
                          allow_patterns=["*.json", "*.py", "*.bin", "*.safetensors", "*.model"])


def audio_features(rows, name, folder, backbone, mert, device, shared=None):
    """Resume partial rows within this new feature batch; never read legacy files."""
    from emomm.features.clamp3 import encode_audio
    unique = list({r["identity"]: r for r in rows}.values())
    positions = {r["identity"]: i for i, r in enumerate(unique)}
    paths = [folder / f"{name}.audio.partial.npy", folder / f"{name}.content.partial.npy"]
    if any(p.exists() for p in paths) and not all(p.exists() for p in paths):
        raise RuntimeError(f"{name} 断点文件不完整，请使用新的 feature_id")
    if paths[0].exists():
        values, hashes = [np.load(p, mmap_mode="r+", allow_pickle=False) for p in paths]
        if values.shape != (len(unique), 768) or hashes.shape != (len(unique),):
            raise ValueError("断点与当前数据行数不一致")
    else:
        values = np.lib.format.open_memmap(paths[0], mode="w+", dtype=np.float32, shape=(len(unique), 768))
        hashes = np.lib.format.open_memmap(paths[1], mode="w+", dtype="<U64", shape=(len(unique),))
    try:
        for i, row in enumerate(unique):
            valid = len(str(hashes[i])) == 64 and np.isfinite(values[i]).all() and np.isclose(np.linalg.norm(values[i]), 1, atol=1e-4)
            if not valid:
                if shared is not None and row["identity"] in shared:
                    vector, digest = shared[row["identity"]]
                else:
                    tokens, digest = mert.encode(row["audio_path"])
                    vector = encode_audio(tokens, backbone, device)
                values[i], hashes[i] = vector, digest
            if (i + 1) % 25 == 0 or i + 1 == len(unique):
                values.flush()
                hashes.flush()
                print(f"{name}: audio {i + 1}/{len(unique)}", flush=True)
        indices = np.asarray([positions[r["identity"]] for r in rows])
        return np.array(values[indices]), np.array(hashes[indices])
    finally:
        values.flush()
        hashes.flush()
        del values, hashes


def tag_features(rows, source_config, encode):
    tracks = [MTGTrack(r["id"], r["artist_id"], r["album_id"], r["audio_path"], 0, r["tags"], r["split"]) for r in rows]
    payload = dict(track_ids=np.asarray([r["id"] for r in rows]),
                   index_positions=np.arange(len(rows)), splits=np.asarray([r["split"] for r in rows]),
                   artist_ids=np.asarray([r["artist_id"] for r in rows]))
    for factor, prefixes in (("affect", ("mood/theme---",)), ("preference", ("genre---", "instrument---"))):
        labels, _ = select_supported_labels(tracks, prefixes,
            min_train_tracks=source_config["min_train_tracks_per_label"],
            min_train_artists=source_config["min_train_artists_per_label"])
        if not labels:
            raise ValueError(f"{factor} 没有满足训练集支持数的标签")
        prompts = [tag_prompt(label) for label in labels]
        payload[f"{factor}_labels"] = np.asarray(labels)
        payload[f"{factor}_prompts"] = np.asarray(prompts)
        payload[f"{factor}_label_embeddings"] = encode(prompts)
        payload[f"{factor}_targets"] = np.asarray([[label in r["tags"] for label in labels] for r in rows], dtype=bool)
    return payload


def extract_lyric_views(config, rows, folder, summary, encode):
    """Add only freshly computed training views, bound to this batch's base file."""
    from emomm.features.lyrics import VIEW_METHOD, VIEW_PARAMETERS, build_lyric_views
    base_path = folder / "pairs_merge.npz"
    expected = summary["files"]["pairs_merge"]["sha256"]
    if sha256(base_path) != expected:
        raise ValueError("MERGE 基础特征已改变，不能生成歌词片段")
    with np.load(base_path, allow_pickle=False) as archive:
        base = {key: archive[key] for key in archive.files}
    if (str(base["schema"]) != SCHEMA or str(base["feature_id"]) != summary["feature_id"]
            or str(base["checkpoint_sha256"]) != config["clamp3"]["checkpoint_sha256"]):
        raise ValueError("MERGE 基础特征的批次或预训练来源不一致")
    payload, counts = build_lyric_views(rows, base, encode)
    payload.update(schema=np.asarray(SCHEMA), feature_id=np.asarray(summary["feature_id"]),
                   checkpoint_sha256=np.asarray(config["clamp3"]["checkpoint_sha256"]),
                   base_sha256=np.asarray(expected))
    output = folder / "pairs_merge_views.npz"
    if output.exists():
        raise FileExistsError(f"歌词片段特征已存在，拒绝覆盖: {output}")
    save_npz(output, **payload)
    summary["files"]["pairs_merge_views"] = dict(name=output.name, sha256=sha256(output))
    summary.setdefault("derived_features", {})["pairs_merge_views"] = dict(
        method=VIEW_METHOD, parameters=VIEW_PARAMETERS, base_sha256=expected, **counts,
        source_text_sha256=fingerprint([(r["id"], r["split"], r["text"]) for r in rows]),
        source_code={str(p.relative_to(ROOT)): sha256(p) for p in
                     (Path(__file__), ROOT / "src/emomm/features/lyrics.py",
                      ROOT / "src/emomm/features/clamp3.py")})
    write_json(resolve(config["outputs"]["records"]) / summary["feature_id"] / "summary.json", summary)
    print(f"歌词片段完成: {counts['eligible_rows']} 首，{counts['selected_views']} 个训练片段", flush=True)


def extract(config, rows_by_name, report, feature_id, resume=False):
    from emomm.features.clamp3 import Clamp3TextAudioEncoder, encode_texts
    from emomm.features.mert import MERTEncoder
    folder = resolve(config["outputs"]["features"]) / feature_id
    records = resolve(config["outputs"]["records"]) / feature_id
    summary_file = records / "summary.json"
    # Both the source metadata and audio file sizes/mtimes bind a resumed run.
    code_files = [Path(__file__), ROOT / "src/emomm/data.py", ROOT / "src/emomm/__init__.py",
                  *sorted((ROOT / "src/emomm/features").glob("*.py"))]
    source_code = {str(p.relative_to(ROOT)): sha256(p) for p in code_files}
    signature = fingerprint({"config": config, "rows": rows_by_name,
                             "source_files": report["source_files"], "source_code": source_code})
    if folder.exists() or records.exists():
        if not resume or not summary_file.is_file():
            raise FileExistsError(f"输出已存在；继续同一次提取需 --resume --feature-id {feature_id}")
        summary = json.loads(summary_file.read_text(encoding="utf-8"))
        if summary["signature"] != signature or summary["schema"] != SCHEMA:
            raise ValueError("源数据或配置已改变，请创建新的 feature_id")
        for entry in summary["files"].values():
            if sha256(folder / entry["name"]) != entry["sha256"]:
                raise ValueError("已提取文件被修改，不能继续使用该断点")
        if summary["status"] == "complete":
            print(f"该提取批次已经完成: {folder}")
            return
    else:
        folder.mkdir(parents=True)
        records.mkdir(parents=True)
        summary = dict(schema=SCHEMA, feature_id=feature_id, signature=signature,
                       status="running", files={}, source_report=report, source_code=source_code)
        save_yaml(records / "effective_config.yaml", config)
    summary["status"] = "running"
    summary.pop("error", None)
    write_json(summary_file, summary)
    try:
        device = require_device(config["device"])
        clamp = config["clamp3"]
        backbone = Clamp3TextAudioEncoder(resolve(clamp["checkpoint"]), text_model_name=clamp["text_model"],
            max_audio_length=clamp["max_audio_length"], normalize=False,
            local_files_only=config["local_files_only"]).to(device).eval()
        tokenizer = backbone.tokenizer(clamp["text_model"], config["local_files_only"])
        encode = lambda texts: encode_texts(texts, backbone, tokenizer, device,
                                            clamp["batch_size_text"], clamp["max_text_length"])
        mert = None
        names = [n for n in DATASETS if n in rows_by_name or n == "mtg_tags" and n in config["datasets"]]
        # Extract MTG before Song Describer so shared songs reuse this run's fresh vectors.
        for name in names:
            if name in summary["files"]:
                continue
            output = folder / f"{name}.npz"
            if name == "mtg_tags":
                payload = tag_features(rows_by_name["mtg_music"], config["sources"]["mtg"], encode)
            else:
                rows = rows_by_name[name]
                payload = metadata_arrays(rows)
                if name.startswith(("text_", "pairs_")):
                    payload["text_embeddings"] = encode([r["text"] for r in rows])
                if any(r["audio_path"] for r in rows):
                    if mert is None:
                        mert = MERTEncoder(config["mert"], device, config["local_files_only"])
                    shared = None
                    if name == "pairs_song_describer" and "mtg_music" in summary["files"]:
                        with np.load(folder / "mtg_music.npz", allow_pickle=False) as data:
                            shared = dict(zip(data["global_identities"], zip(data["embeddings"], data["audio_content_hashes"])))
                    audio, hashes = audio_features(rows, name, folder, backbone, mert, device, shared)
                    payload["audio_embeddings" if name != "mtg_music" else "embeddings"] = audio
                    payload["audio_content_hashes"] = hashes
                if name == "mtg_music":
                    payload.update(track_ids=np.asarray([r["id"] for r in rows]),
                                   audio_paths=np.asarray([r["audio_path"] for r in rows]))
            payload.update(schema=np.asarray(SCHEMA), feature_id=np.asarray(feature_id),
                           checkpoint_sha256=np.asarray(clamp["checkpoint_sha256"]))
            save_npz(output, **payload)
            summary["files"][name] = dict(name=output.name, sha256=sha256(output))
            write_json(summary_file, summary)
            # Only this run's known intermediate files are removed after the NPZ is committed.
            for suffix in ("audio.partial.npy", "content.partial.npy"):
                (folder / f"{name}.{suffix}").unlink(missing_ok=True)
            print(f"完成: {output}", flush=True)
            del payload
            gc.collect()
        if "pairs_merge" in summary["files"] and "pairs_merge_views" not in summary["files"]:
            # Audio extraction has finished; free MERT before encoding the lyric views.
            mert = None
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
            extract_lyric_views(config, rows_by_name["pairs_merge"], folder, summary, encode)
        summary["status"] = "complete"
        write_json(summary_file, summary)
        print(f"提取完成。feature_id={feature_id}\n特征: {folder}\n运行记录: {records}")
    except BaseException as error:
        summary.update(status="failed", error=repr(error))
        write_json(summary_file, summary)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "config/data.yaml")
    parser.add_argument("--feature-id")
    parser.add_argument("--datasets", nargs="+", choices=DATASETS)
    parser.add_argument("--device", choices=("cpu", "cuda"))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    config = load_yaml(resolve(args.config))
    if args.datasets:
        config["datasets"] = args.datasets
    if "mtg_tags" in config["datasets"] and "mtg_music" not in config["datasets"]:
        config["datasets"].append("mtg_music")
    if args.device:
        config["device"] = args.device
    require_device(config["device"])
    if args.resume and not args.feature_id:
        parser.error("--resume 需要明确 --feature-id")
    selected_id = run_id(args.feature_id or config["feature_id"])
    config["feature_id"] = selected_id
    rows, report = load_raw(config)
    check_pretrained(config)
    print(json.dumps({"datasets": report["datasets"], "missing_audio": {k: len(v) for k, v in report["missing_audio"].items()}}, ensure_ascii=False, indent=2))
    if args.check_only:
        print("检查通过；未生成特征或运行产物。")
        return
    extract(config, rows, report, selected_id, args.resume)


if __name__ == "__main__":
    main()
