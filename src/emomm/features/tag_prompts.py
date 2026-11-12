"""Label-preserving MTG text views; audio features and target labels stay fixed."""
from __future__ import annotations

import gc
import json
from pathlib import Path

import numpy as np

from .. import SCHEMA, fingerprint, require_device, resolve, save_npz
from ..data import read_features
from ..train import load_yaml, sha256, write_json


PROMPT_METHOD = "mtg_label_preserving_prompt_views_v1"
TRAIN_VIEWS = 4
FACTORS = ("affect", "preference")


def prompt_views(canonical: str) -> tuple[list[str], str]:
    """Use grammatical wrappers, without adding synonyms or musical conditions.

    The exact cached canonical prompt is view zero, including any inherited
    wording. The other training views and the held-out evaluation template
    contain precisely the same musical phrase. No song metadata is consulted.
    """
    phrase = str(canonical).strip().rstrip(".")
    if not phrase:
        raise ValueError("A label prompt cannot be empty")
    training = [str(canonical), f"A piece of {phrase}.",
                f"I would like to listen to {phrase}.", f"Please recommend {phrase}."]
    held_out = f"Find me a track described as {phrase}."
    if len(set(training + [held_out])) != TRAIN_VIEWS + 1:
        raise ValueError("Prompt views must be distinct")
    return training, held_out


def _specification(data_config, feature_id, folder, base_sha256):
    base_path = Path(folder) / "mtg_tags.npz"
    if sha256(base_path) != base_sha256:
        raise ValueError("MTG tag features changed before prompt preparation")
    base = read_features(base_path)
    extraction_config = load_yaml(resolve(data_config["outputs"]["records"]) / feature_id / "effective_config.yaml")
    original = extraction_config["clamp3"]
    current = data_config["clamp3"]
    for key in ("checkpoint_sha256", "text_model", "embedding_dim", "max_text_length"):
        if current[key] != original[key]:
            raise ValueError(f"New tag prompts must use the original feature batch's clamp3.{key}")
    if str(base["feature_id"]) != feature_id or str(base["checkpoint_sha256"]) != current["checkpoint_sha256"]:
        raise ValueError("MTG prompt source does not belong to this feature batch")
    factors = {}
    for factor in FACTORS:
        labels = base[f"{factor}_labels"].astype(str).tolist()
        canonical = base[f"{factor}_prompts"].astype(str).tolist()
        if not labels or len(set(labels)) != len(labels) or len(labels) != len(canonical):
            raise ValueError(f"Invalid {factor} label vocabulary")
        views = [prompt_views(prompt) for prompt in canonical]
        factors[factor] = dict(labels=labels, training_prompts=[item[0] for item in views],
                               held_out_prompts=[item[1] for item in views])
    specification = dict(method=PROMPT_METHOD, feature_id=feature_id, base_sha256=base_sha256,
                         text_encoder={key: current[key] for key in
                                       ("checkpoint_sha256", "text_model", "embedding_dim", "max_text_length")},
                         training_views_per_label=TRAIN_VIEWS,
                         training_sampling="one_uniform_view_per_label_per_step_with_private_rng",
                         canonical_embedding="exact_copy_of_original_cached_vector",
                         vocabulary_source="existing_train_supported_mtg_labels",
                         song_targets="unchanged_original_multilabel_matrix",
                         held_out_usage="final_validation_only_never_training_or_checkpoint_selection",
                         factors=factors)
    return base, specification


def validate_prompt_cache(values, base, specification, signature):
    if (str(values.get("prompt_method", "")) != PROMPT_METHOD
            or str(values.get("signature", "")) != signature
            or str(values.get("feature_id", "")) != specification["feature_id"]
            or str(values.get("checkpoint_sha256", "")) != specification["text_encoder"]["checkpoint_sha256"]
            or str(values.get("base_sha256", "")) != specification["base_sha256"]
            or str(values.get("specification_json", "")) != json.dumps(specification, ensure_ascii=False, sort_keys=True)):
        raise ValueError("Tag prompt cache provenance does not match this feature batch")
    for factor in FACTORS:
        spec = specification["factors"][factor]
        labels = np.asarray(spec["labels"])
        prompts = np.asarray(spec["training_prompts"])
        if (not np.array_equal(values[f"{factor}_labels"], labels)
                or not np.array_equal(values[f"{factor}_view_prompts"], prompts)
                or not np.array_equal(values[f"{factor}_held_out_prompts"], spec["held_out_prompts"])):
            raise ValueError(f"Tag prompt cache has changed {factor} label order or wording")
        views = values[f"{factor}_view_embeddings"]
        held_out = values[f"{factor}_held_out_embeddings"]
        if views.shape != (len(labels) * TRAIN_VIEWS, 768) or held_out.shape != (len(labels), 768):
            raise ValueError(f"Invalid {factor} prompt view dimensions")
        for vectors in (views, held_out):
            if not np.isfinite(vectors).all() or not np.allclose(np.linalg.norm(vectors, axis=1), 1, atol=1e-4):
                raise ValueError(f"Invalid or unnormalized {factor} prompt vectors")
        if not np.array_equal(views.reshape(len(labels), TRAIN_VIEWS, 768)[:, 0], base[f"{factor}_label_embeddings"]):
            raise ValueError("Canonical prompt vectors must exactly reuse the original cache")


def prepare_prompt_features(data_config, feature_id, folder, base_sha256, device, *, check_only=False):
    """Lazily encode only new label text, caching it beside the immutable base NPZ.

    A supplementary record has its own hash; the completed extraction summary
    and its original files are never rewritten. Check-only creates no files.
    """
    base, specification = _specification(data_config, feature_id, folder, base_sha256)
    signature = fingerprint(specification)
    stem = f"mtg_tag_prompt_views_{signature[:16]}"
    path = Path(folder) / f"{stem}.npz"
    record = resolve(data_config["outputs"]["records"]) / feature_id / f"{stem}.json"
    info = dict(path=str(path), record=str(record), method=PROMPT_METHOD, signature=signature,
                training_views_per_label=TRAIN_VIEWS,
                labels={f: len(specification["factors"][f]["labels"]) for f in FACTORS},
                new_texts=sum(len(specification["factors"][f]["labels"]) * TRAIN_VIEWS for f in FACTORS))
    cached = path.is_file()
    if cached:
        values = read_features(path)
        validate_prompt_cache(values, base, specification, signature)
        info["sha256"] = sha256(path)
        if record.is_file():
            previous = json.loads(record.read_text(encoding="utf-8"))
            if previous.get("sha256") != info["sha256"] or previous.get("specification") != specification:
                raise ValueError("Tag prompt extraction record disagrees with the cache")
    info["status"] = "cached" if cached else "pending_text_encoding"
    if check_only:
        return info
    if not cached:
        import torch
        from .clamp3 import Clamp3TextAudioEncoder, encode_texts

        cfg = data_config["clamp3"]
        checkpoint = resolve(cfg["checkpoint"])
        if sha256(checkpoint) != cfg["checkpoint_sha256"]:
            raise ValueError("CLaMP3 checkpoint hash verification failed")
        selected_device = require_device(device)
        print(f"补提取 A/P 标签文本: {info['new_texts']} 条；复用原音频和原始提示向量", flush=True)
        model = Clamp3TextAudioEncoder(checkpoint, text_model_name=cfg["text_model"],
            max_audio_length=cfg["max_audio_length"], local_files_only=data_config["local_files_only"])
        # This step needs only CLaMP3's text branch; do not move audio weights to the GPU.
        model.text_model.to(selected_device)
        model.text_proj.to(selected_device)
        tokenizer = model.tokenizer(cfg["text_model"], data_config["local_files_only"])
        try:
            texts = []
            for factor in FACTORS:
                spec = specification["factors"][factor]
                texts.extend(text for views in spec["training_prompts"] for text in views[1:])
                texts.extend(spec["held_out_prompts"])
            encoded = encode_texts(texts, model, tokenizer, selected_device,
                                   cfg["batch_size_text"], cfg["max_text_length"])
        finally:
            del model, tokenizer
            gc.collect()
            if selected_device.type == "cuda":
                torch.cuda.empty_cache()
        payload = dict(schema=np.asarray(SCHEMA), feature_id=np.asarray(feature_id),
                       checkpoint_sha256=np.asarray(cfg["checkpoint_sha256"]),
                       base_sha256=np.asarray(base_sha256), prompt_method=np.asarray(PROMPT_METHOD),
                       signature=np.asarray(signature),
                       specification_json=np.asarray(json.dumps(specification, ensure_ascii=False, sort_keys=True)))
        cursor = 0
        for factor in FACTORS:
            spec = specification["factors"][factor]
            count = len(spec["labels"])
            views = np.empty((count, TRAIN_VIEWS, 768), dtype=np.float32)
            views[:, 0] = base[f"{factor}_label_embeddings"]
            views[:, 1:] = encoded[cursor:cursor + count * (TRAIN_VIEWS - 1)].reshape(count, TRAIN_VIEWS - 1, 768)
            cursor += count * (TRAIN_VIEWS - 1)
            payload[f"{factor}_labels"] = np.asarray(spec["labels"])
            payload[f"{factor}_view_prompts"] = np.asarray(spec["training_prompts"])
            payload[f"{factor}_view_embeddings"] = views.reshape(count * TRAIN_VIEWS, 768)
            payload[f"{factor}_held_out_prompts"] = np.asarray(spec["held_out_prompts"])
            payload[f"{factor}_held_out_embeddings"] = encoded[cursor:cursor + count]
            cursor += count
        if cursor != len(encoded):
            raise RuntimeError("Tag prompt encoding row accounting mismatch")
        validate_prompt_cache(payload, base, specification, signature)
        save_npz(path, **payload)
        validate_prompt_cache(read_features(path), base, specification, signature)
        info.update(status="encoded", sha256=sha256(path))
    if not record.is_file():
        write_json(record, dict(schema=SCHEMA, status="complete", name=path.name,
                                sha256=info["sha256"], specification=specification))
    return info
