"""Protect label identity, immutable caches and matched prompt comparison runs."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from emomm import SCHEMA, save_npz, save_yaml
from emomm.features.tag_prompts import prepare_prompt_features, TRAIN_VIEWS
from emomm.models.three_branch import FrozenCLaMP3Retriever
from emomm.train import load_yaml, sha256
from emomm.train.three_branch import sample_label_views, tag_evaluation, train_comparison


def normalized(rng, count):
    values = rng.standard_normal((count, 768)).astype(np.float32)
    return values / np.linalg.norm(values, axis=1, keepdims=True)


class TagPromptComparisonTests(unittest.TestCase):
    def test_lazy_cache_preserves_canonical_vectors_and_rejects_changed_provenance(self):
        rng = np.random.default_rng(7)
        with tempfile.TemporaryDirectory(prefix="emomm_prompt_cache_") as temp:
            root = Path(temp)
            folder = root / "features"
            records = root / "records"
            folder.mkdir()
            checkpoint = root / "checkpoint.pt"
            checkpoint.write_bytes(b"test checkpoint; never passed to a real encoder")
            data_config = dict(clamp3=dict(checkpoint=str(checkpoint), checkpoint_sha256=sha256(checkpoint),
                text_model="test-model", embedding_dim=768, max_text_length=128,
                max_audio_length=128, batch_size_text=32),
                local_files_only=True, outputs=dict(records=str(records)))
            save_yaml(records / "test_batch/effective_config.yaml", data_config)
            base = dict(schema=np.asarray(SCHEMA), feature_id=np.asarray("test_batch"),
                        checkpoint_sha256=np.asarray(sha256(checkpoint)))
            for factor in ("affect", "preference"):
                base[f"{factor}_labels"] = np.asarray([f"{factor}:0", f"{factor}:1"])
                base[f"{factor}_prompts"] = np.asarray(["sad music", "music featuring piano"])
                base[f"{factor}_label_embeddings"] = normalized(rng, 2)
            save_npz(folder / "mtg_tags.npz", **base)
            digest = sha256(folder / "mtg_tags.npz")
            before = sorted(str(path.relative_to(root)) for path in root.rglob("*"))
            check = prepare_prompt_features(data_config, "test_batch", folder, digest, "cpu", check_only=True)
            self.assertEqual(check["status"], "pending_text_encoding")
            self.assertEqual(before, sorted(str(path.relative_to(root)) for path in root.rglob("*")))
            encoder = SimpleNamespace(text_model=torch.nn.Identity(), text_proj=torch.nn.Identity(),
                                      tokenizer=lambda *args: None)
            def encode(texts, *args):
                return normalized(rng, len(texts))
            with patch("emomm.features.clamp3.Clamp3TextAudioEncoder", return_value=encoder), \
                 patch("emomm.features.clamp3.encode_texts", side_effect=encode) as encode_call:
                info = prepare_prompt_features(data_config, "test_batch", folder, digest, "cpu")
                cached = prepare_prompt_features(data_config, "test_batch", folder, digest, "cpu")
                self.assertEqual(encode_call.call_count, 1)
            self.assertEqual(info["sha256"], cached["sha256"])
            self.assertEqual(cached["status"], "cached")
            self.assertEqual(digest, sha256(folder / "mtg_tags.npz"))
            with np.load(info["path"], allow_pickle=False) as values:
                for factor in ("affect", "preference"):
                    np.testing.assert_array_equal(values[f"{factor}_view_embeddings"].reshape(2, TRAIN_VIEWS, 768)[:, 0],
                                                  base[f"{factor}_label_embeddings"])
                    train_texts = set(values[f"{factor}_view_prompts"].ravel())
                    self.assertTrue(train_texts.isdisjoint(values[f"{factor}_held_out_prompts"]))
            report_path = Path(info["record"])
            report = json.loads(report_path.read_text(encoding="utf-8"))
            report["sha256"] = "changed"
            report_path.write_text(json.dumps(report), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "record disagrees"):
                prepare_prompt_features(data_config, "test_batch", folder, digest, "cpu", check_only=True)

    def test_sampling_preserves_label_rows_without_consuming_training_rng(self):
        views = torch.arange(6 * TRAIN_VIEWS * 768).reshape(6, TRAIN_VIEWS, 768).float()
        torch.manual_seed(91)
        np.random.seed(91)
        torch_state = torch.random.get_rng_state().clone()
        numpy_state = np.random.get_state()
        first = sample_label_views(views, np.random.default_rng(42))
        second = sample_label_views(views, np.random.default_rng(42))
        self.assertTrue(torch.equal(first, second))
        self.assertEqual(tuple(first.shape), (6, 768))
        for label in range(6):
            self.assertTrue(any(torch.equal(first[label], view) for view in views[label]))
        self.assertTrue(torch.equal(torch_state, torch.random.get_rng_state()))
        for old, new in zip(numpy_state, np.random.get_state()):
            np.testing.assert_array_equal(old, new)

    def test_end_to_end_four_methods_keep_test_rows_unused_and_primary_queries_fixed(self):
        rng = np.random.default_rng(19)
        music = normalized(rng, 15)
        splits = np.asarray(["train"] * 9 + ["validation"] * 5 + ["test"])
        pairs = dict(text_embeddings=normalized(rng, 15), audio_embeddings=music.copy(),
                     splits=splits.copy(), datasets=np.asarray(["synthetic"] * 15),
                     global_identities=np.asarray([f"track:{i}" for i in range(15)]))
        tags = dict(splits=splits, index_positions=np.arange(15))
        for factor in ("affect", "preference"):
            canonical = normalized(rng, 4)
            targets = np.zeros((15, 4), dtype=bool)
            targets[np.arange(15), np.arange(15) % 4] = True
            targets[::3, 1] = True
            targets[1] = False  # A batch may contain no positive for a query/track.
            tags[f"{factor}_label_embeddings"] = canonical
            tags[f"{factor}_view_embeddings"] = np.stack([canonical, *[normalized(rng, 4) for _ in range(3)]], axis=1).reshape(-1, 768)
            tags[f"{factor}_held_out_embeddings"] = normalized(rng, 4)
            tags[f"{factor}_targets"] = targets
        # NaNs in test-only rows make accidental test use fail immediately.
        music[-1] = np.nan
        pairs["audio_embeddings"][-1] = np.nan
        pairs["text_embeddings"][-1] = np.nan
        before = tag_evaluation(FrozenCLaMP3Retriever(), "affect", tags, music, "validation", "cpu", 3, (5, 10, 20))
        changed = copy.deepcopy(tags)
        changed["affect_view_embeddings"][:] = np.nan
        after = tag_evaluation(FrozenCLaMP3Retriever(), "affect", changed, music, "validation", "cpu", 3, (5, 10, 20))
        self.assertEqual(before, after)
        config = copy.deepcopy(load_yaml(ROOT / "config/train.yaml")["three_branch"])
        config.update(device="cpu", run_id="synthetic", feature_id="synthetic")
        config["model"]["bottleneck_dim"] = 8
        config["training"].update(max_epochs=2, batch_size_pairs=3, batch_size_mtg=3, amp=False)
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory(prefix="emomm_prompt_training_") as temp:
            output = Path(temp)
            selected = train_comparison(config, pairs, tags, music, [42, 43], output)
            self.assertTrue(all(row["method"] == "residual_distill" for row in selected))
            report = json.loads((output / "comparison.json").read_text(encoding="utf-8"))
            self.assertEqual(set(report["methods"]), {"clamp3", "adapter", "residual_distill", "ap_multiview"})
            comparison = json.loads((output / "ap_prompt_comparison.json").read_text(encoding="utf-8"))
            self.assertEqual(len(comparison["rows"]), 8)
            for row in comparison["rows"]:
                self.assertAlmostEqual(row["absolute_delta"], row["multiview_mean"] - row["single_mean"])
                self.assertEqual(row["total_seeds"], 2)
            for method, target in (("residual_distill", output), ("ap_multiview", output / "comparison/ap_multiview")):
                for seed in (42, 43):
                    metrics = json.loads((target / f"seed_{seed}/metrics.json").read_text(encoding="utf-8"))
                    self.assertEqual(set(metrics["results"]), {"validation"})
                    self.assertTrue(metrics["training"]["save_reload_verified"])
                    self.assertEqual(metrics["training"]["ap_prompt_training"]["views_per_label"], TRAIN_VIEWS if method == "ap_multiview" else 1)


if __name__ == "__main__":
    unittest.main()
