"""Versioned banks/tasks with a physical boundary between input and test labels."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import numpy as np
import yaml

from emomm_sequence.losses.sequence_cost import normalize, make_problem
from emomm_sequence.training.records import sha256, write_json, check_identity_splits, environment

ROOT = Path(__file__).resolve().parents[3]


def load_config(path):
    config = yaml.safe_load((ROOT / path).read_text(encoding="utf-8"))
    return config, ROOT / config["output_root"]


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".jsonl.tmp")
    with temporary.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


@dataclass
class Bank:
    source: str
    split: str
    ids: np.ndarray
    groups: np.ndarray
    x: np.ndarray

    def subset(self, indices):
        ix = np.arange(len(self.ids)) if indices is None else np.asarray(indices, dtype=int)
        return Bank(self.source, self.split, self.ids[ix], self.groups[ix], self.x[ix])


def load_bank(root, source, split):
    # No labels are available through this API.
    with np.load(Path(root) / "banks" / f"{source}_{split}.npz", allow_pickle=False) as z:
        if set(z.files) != {"track_ids", "content_group_ids", "embeddings"}:
            raise ValueError("bank contains unexpected fields (possible label leak)")
        return Bank(source, split, z["track_ids"], z["content_group_ids"], z["embeddings"])


def make_task_problem(bank, task, text_bank):
    selected = bank.subset(task.get("indices"))
    vectors = text_bank["embeddings"]
    pref = None if task["preference_text_id"] is None else vectors[task["preference_text_id"]]
    return selected, make_problem(selected.x, selected.ids, vectors[task["start_text_id"]],
                                  vectors[task["target_text_id"]], pref, int(task["num_tracks"]))


def prepare_banks(config, root):
    (root / "banks").mkdir(parents=True, exist_ok=True)
    (root / "labels").mkdir(parents=True, exist_ok=True)
    assets = config["assets"]
    report = {"sources": {}, "input_hashes": {}, "scope": "content-group split; artist separation audited separately"}
    all_groups, all_splits = [], []
    with np.load(ROOT / assets["mtg_original_embeddings"], allow_pickle=False) as z:
        catalog_ids, catalog_x = z["track_ids"].astype(str), z["embeddings"]
    lookup = {x: i for i, x in enumerate(catalog_ids)}
    if len(lookup) != len(catalog_ids):
        raise ValueError("duplicate catalog IDs")
    report["input_hashes"][assets["mtg_original_embeddings"]] = sha256(ROOT / assets["mtg_original_embeddings"])
    for source in ("deam", "pmemo", "merge", "mtg"):
        relative = assets["mtg_labels"] if source == "mtg" else assets[source]
        report["input_hashes"][relative] = sha256(ROOT / relative)
        with np.load(ROOT / relative, allow_pickle=False) as z:
            splits, groups = z["splits"].astype(str), z["content_group_ids"].astype(str)
            ids = z["track_ids"].astype(str) if source == "mtg" else z["global_identities"].astype(str)
            if source == "mtg":
                positions = np.array([lookup[i] for i in ids])
                x = catalog_x[positions]
                labels = {"preference_targets": z["preference_targets"], "preference_labels": z["preference_labels"],
                          "affect_targets": z["affect_targets"], "affect_labels": z["affect_labels"]}
                train = splits == "train"
                label_names = z["preference_labels"].astype(str)
                counts = z["preference_targets"][train].sum(axis=0)
                prefs = []
                for kind in ("genre", "instrument"):
                    candidates = [i for i, name in enumerate(label_names) if name.startswith(kind + "---")]
                    candidates.sort(key=lambda i: (-int(counts[i]), label_names[i]))
                    prefs.extend([{"label": str(label_names[i]), "label_index": int(i), "train_count": int(counts[i]),
                                   "kind": kind, "phrase": str(label_names[i]).split("---", 1)[1]} for i in candidates[:4]])
                if len(prefs) != 8:
                    raise ValueError("need four genre and four instrument labels")
                report["preferences"] = prefs
                overlap = {}
                for field in ("artist_ids", "album_ids"):
                    values = z[field].astype(str)
                    train_values = set(values[train])
                    overlap[field] = {split: float(np.isin(values[splits == split], list(train_values)).mean()) for split in ("validation", "test")}
                report["mtg_train_overlap"] = overlap
            else:
                x = z["audio_embeddings"]
                va = (z["va_targets"].astype(np.float32) + 1) / 2
                if not np.isfinite(va).all() or np.any((va < 0) | (va > 1)):
                    raise ValueError(f"invalid VA labels in {source}")
                labels = {"va": va}
        check_identity_splits(groups, splits, source)
        all_groups.extend(groups.tolist())
        all_splits.extend(splits.tolist())
        x = normalize(x)
        counts = {}
        for split in ("train", "validation", "test", "all"):
            raw = np.flatnonzero(splits == split) if split != "all" else np.arange(len(splits))
            # One song per content group, with canonical ID deciding its representative.
            selected = {}
            for i in sorted(raw, key=lambda i: ids[i]):
                selected.setdefault(groups[i], int(i))
            ix = np.array(list(selected.values()), dtype=np.int64)
            if len(set(ids[ix])) != len(ix):
                raise ValueError(f"duplicate IDs across content groups: {source}")
            np.savez_compressed(root / "banks" / f"{source}_{split}.npz", track_ids=ids[ix], content_group_ids=groups[ix], embeddings=x[ix])
            output_labels = {"track_ids": ids[ix]}
            for name, values in labels.items():
                output_labels[name] = values if name.endswith("_labels") else values[ix]
            np.savez_compressed(root / "labels" / f"{source}_{split}.npz", **output_labels)
            counts[split] = {"rows": len(raw), "unique_content": len(ix)}
        report["sources"][source] = counts
    check_identity_splits(np.asarray(all_groups), np.asarray(all_splits), "cross-source content")
    report["cross_split_content_conflicts"] = 0
    write_json(root / "protocol" / "asset_audit.json", report)
    return report


def prepare_text(config, root, audit, device):
    prompts, mapping = [], {}
    def add(text):
        if text not in mapping:
            mapping[text] = len(prompts)
            prompts.append(text)
        return mapping[text]
    states = []
    tp = config["text_protocol"]
    for name, state in tp["states"].items():
        states.append({"name": name, "va": state["target_va"], "text_ids": [add(tp["emotion_template"].format(phrase=p)) for p in state["phrases"]]})
    prefs = []
    for pref in audit["preferences"]:
        template = tp["genre_template"] if pref["kind"] == "genre" else tp["instrument_template"]
        prefs.append({**pref, "text_id": add(template.format(preference=pref["phrase"]))})
    key = hashlib.sha256(json.dumps({"prompts": prompts, "backbone": config["assets"]["backbone_sha256"]}, sort_keys=True).encode()).hexdigest()
    cache = root / "cache" / "text_bank.npz"
    cache.parent.mkdir(parents=True, exist_ok=True)
    if cache.exists():
        with np.load(cache, allow_pickle=False) as z:
            if str(z["cache_key"].item()) != key:
                raise ValueError("existing text cache differs from frozen prompts/checkpoint")
    else:
        import torch
        from emomm_sequence.encoding.text import FrozenTextEncoder
        model = FrozenTextEncoder(ROOT / config["assets"]["backbone"], "FacebookAI/xlm-roberta-base").to(device).eval()
        tokenizer = model.tokenizer()
        embeddings = []
        for i in range(0, len(prompts), 8):
            tokens = tokenizer(prompts[i:i+8], padding=True, truncation=True, max_length=128, return_tensors="pt")
            embeddings.append(model.encode(tokens.input_ids.to(device), tokens.attention_mask.to(device)).cpu().numpy())
        np.savez_compressed(cache, texts=np.asarray(prompts), embeddings=np.concatenate(embeddings), cache_key=np.array(key))
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    metadata = {"states": states, "preferences": prefs, "prompts": prompts, "cache_key": key}
    write_json(root / "protocol" / "text_protocol.json", metadata)
    return metadata


def prepare_tasks(config, root, profile, meta):
    pc = config["profiles"][profile]
    out = root / "tasks" / profile
    out.mkdir(parents=True, exist_ok=True)
    fingerprint = hashlib.sha256(json.dumps({"config": config, "text": meta, "profile": profile}, sort_keys=True).encode()).hexdigest()
    stamp = out / "manifest.json"
    if stamp.exists():
        old = json.loads(stamp.read_text(encoding="utf-8"))
        if old["fingerprint"] != fingerprint:
            raise ValueError("task configuration changed; use a new output_root instead of overwriting")
        for name, digest in old["files"].items():
            if sha256(out / name) != digest:
                raise ValueError(f"task file modified: {name}")
        return old
    bank_sizes = {(s, p): len(load_bank(root, s, p).ids) for s in ("deam", "pmemo", "merge", "mtg") for p in ("train", "validation", "test")}
    rng = np.random.default_rng(20260929)
    def request(source, split, a, b, phrase, n, budget, suite, family, pref=None, indices=None):
        return {"source": source, "split": split, "suite": suite, "query_family_id": family,
                "start_text_id": meta["states"][a]["text_ids"][phrase], "target_text_id": meta["states"][b]["text_ids"][phrase],
                "preference_text_id": None if pref is None else pref["text_id"],
                "preference_label_index": None if pref is None else pref["label_index"],
                "start_va": meta["states"][a]["va"], "target_va": meta["states"][b]["va"],
                "num_tracks": int(n), "budget": int(budget), "indices": indices, "phrase_index": phrase}
    files = {}
    for split, count in (("train", pc.get("train_tasks", pc.get("rl_tasks"))), ("validation", pc["validation_tasks"])):
        rows = []
        for i in range(count):
            source = ("deam", "pmemo", "merge", "mtg")[i % 4]
            a, b = rng.integers(0, 4, size=2)
            n = int(rng.choice(config["training"]["episode_num_tracks"]))
            budget = int(rng.choice(config["training"]["episode_candidate_budgets"]))
            size = min(config["training"]["episode_library_size"], bank_sizes[(source, split)])
            indices = sorted(rng.choice(bank_sizes[(source, split)], size=size, replace=False).tolist())
            pref = meta["preferences"][int(rng.integers(8))] if source == "mtg" else None
            rows.append(request(source, split, int(a), int(b), int(rng.integers(2)), n, min(budget, size), "episode", f"{split}_{i}", pref, indices))
        for i, r in enumerate(rows):
            r["query_id"] = f"{profile}_{split}_{i:06d}"
        name = f"{split}.jsonl"
        write_jsonl(out / name, rows)
        files[name] = sha256(out / name)
    for split in ("validation", "test"):
        rows = []
        e1_count = pc.get("e1_query_families", config["evaluation"]["e1"]["query_families"])
        for f in range(e1_count):
            source = ("deam", "pmemo", "merge", "mtg")[f % 4]
            indices = sorted(rng.choice(bank_sizes[(source, split)], size=16, replace=False).tolist())
            a, b = rng.integers(0, 4, size=2)
            for n in (3, 5, 7):
                for budget in (8, 12, 16):
                    rows.append(request(source, split, int(a), int(b), 2, n, budget, "e1", f"e1_{source}_{f}", indices=indices))
        pairs = [(a, b) for a in range(4) for b in range(4)]
        e2_pairs = pairs if profile == "full" else [pairs[i] for i in sorted(rng.choice(16, pc["e2_query_families_per_source"], replace=False))]
        for source in ("deam", "pmemo", "merge"):
            for a, b in e2_pairs:
                for phrase in range(3):
                    for n in config["evaluation"]["e2"]["num_tracks"]:
                        for budget in config["evaluation"]["e2"]["candidate_budgets"]:
                            rows.append(request(source, split, a, b, phrase, n, budget, "e2", f"e2_{a}_{b}"))
        families = [(a, b, p) for a, b in pairs if a != b for p in range(8)]
        if profile == "pilot":
            families = [families[i] for i in sorted(rng.choice(len(families), pc["e3_query_families"], replace=False))]
        for a, b, p in families:
            for n in config["evaluation"]["e3"]["num_tracks"]:
                for budget in config["evaluation"]["e3"]["candidate_budgets"]:
                    rows.append(request("mtg", split, a, b, 2, n, budget, "e3", f"e3_{a}_{b}_{p}", meta["preferences"][p]))
        for i, r in enumerate(rows):
            r["query_id"] = f"{profile}_{split}_eval_{i:06d}"
        name = f"evaluate_{split}.jsonl"
        write_jsonl(out / name, rows)
        files[name] = sha256(out / name)
    result = {"profile": profile, "fingerprint": fingerprint, "files": files, "human_playlist_labels": False}
    write_json(stamp, result)
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/experiments/sequence_routes.yaml")
    p.add_argument("--profile", choices=["pilot", "full"], default="pilot")
    p.add_argument("--audit-only", action="store_true")
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    config, root = load_config(args.config)
    import torch
    torch.set_num_threads(4)
    audit = prepare_banks(config, root)
    write_json(root / "protocol" / "environment.json", environment())
    (root / "protocol" / "resolved_protocol.yaml").write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8")
    if args.audit_only:
        print(json.dumps({"status": "audit_ok", "sources": audit["sources"]}))
        return
    actual = sha256(ROOT / config["assets"]["backbone"])
    if actual != config["assets"]["backbone_sha256"]:
        raise ValueError("CLaMP3 checkpoint hash mismatch")
    meta = prepare_text(config, root, audit, args.device)
    manifest = prepare_tasks(config, root, args.profile, meta)
    print(json.dumps({"status": "prepared", **manifest}))
