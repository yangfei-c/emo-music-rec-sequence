"""Single production implementation of EmoMM-Sequence."""

from __future__ import annotations

import hashlib
import re
import threading
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from .encoder import FrozenTextEncoder
from .relation import RelationEnsemble
from .retriever import FrozenRetriever
from .text_va import load_text_va
from .search import Candidate, direction_from_va, directional_scores, fallback_sequence, select_music_sequence


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def _stable_order(scores: np.ndarray, ids: np.ndarray) -> np.ndarray:
    return np.lexsort((ids.astype(str), -np.asarray(scores, dtype=np.float64)))


class EmoMMSequence:
    """Load frozen assets once and return five-track sequences."""

    def __init__(
        self,
        root: str | Path | None = None,
        config: str | Path = "configs/system.yaml",
        device: str | None = None,
        verify_hashes: bool = True,
    ) -> None:
        self.root = Path(root) if root is not None else Path(__file__).resolve().parents[2]
        config_path = _resolve(self.root, config)
        self.config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        self.config_sha256 = _sha256(config_path)
        requested = device or str(self.config["device"])
        self.device = torch.device(requested if requested != "cuda" or torch.cuda.is_available() else "cpu")
        self._lock = threading.Lock()
        if verify_hashes:
            self._verify_assets()

        backbone = self.config["backbone"]
        self.encoder = FrozenTextEncoder(
            _resolve(self.root, backbone["checkpoint"]), str(backbone["text_model"])
        ).to(self.device).eval()
        self.tokenizer = self.encoder.tokenizer()
        retriever = self.config["retriever"]
        self.retriever = FrozenRetriever(
            _resolve(self.root, retriever["checkpoint"]),
            embedding_dim=int(backbone["embedding_dim"]),
            bottleneck_dim=int(retriever["bottleneck_dim"]),
            dropout=float(retriever["dropout"]),
        ).to(self.device).eval()
        self.relation = RelationEnsemble(self.root, self.config["relation"], self.device)
        text_va = self.config["text_va"]
        self.text_va = load_text_va(
            _resolve(self.root, text_va["checkpoint"]), retriever["checkpoint_sha256"]
        ).to(self.device)
        self._load_catalog()
        self._load_affect_terms()

    def _verify(self, value: str, expected: str) -> None:
        path = _resolve(self.root, value)
        if not path.is_file():
            raise FileNotFoundError(path)
        if _sha256(path) != str(expected):
            raise ValueError(f"asset hash mismatch: {path}")

    def _verify_assets(self) -> None:
        backbone = self.config["backbone"]
        self._verify(backbone["checkpoint"], backbone["checkpoint_sha256"])
        retriever = self.config["retriever"]
        self._verify(retriever["checkpoint"], retriever["checkpoint_sha256"])
        text_va = self.config["text_va"]
        self._verify(text_va["checkpoint"], text_va["checkpoint_sha256"])
        catalog = self.config["catalog"]
        self._verify(catalog["music"], catalog["music_sha256"])
        self._verify(catalog["metadata"], catalog["metadata_sha256"])
        self._verify(catalog["factor_track_ids"], catalog["factor_track_ids_sha256"])
        for value in catalog["factors"].values():
            self._verify(value["path"], value["sha256"])
        affect = self.config["affect_direction"]
        self._verify(affect["public_terms"], affect["public_terms_sha256"])
        relation = self.config["relation"]
        for seed in relation["seeds"]:
            expected_hashes = relation["checkpoints_sha256"]
            expected = expected_hashes.get(seed, expected_hashes.get(str(seed)))
            if expected is None:
                raise KeyError(f"missing relation checkpoint hash for seed {seed}")
            self._verify(
                relation["checkpoint_template"].format(seed=seed),
                expected,
            )
            calibration_hashes = relation["calibration_sha256"]
            calibration_hash = calibration_hashes.get(seed, calibration_hashes.get(str(seed))) if isinstance(calibration_hashes, dict) else calibration_hashes
            self._verify(relation["calibration_template"].format(seed=seed), calibration_hash)

    def _load_catalog(self) -> None:
        catalog = self.config["catalog"]
        with np.load(_resolve(self.root, catalog["music"]), allow_pickle=False) as data:
            self.track_ids = data["track_ids"].astype(str)
            self.audio_embeddings = data["embeddings"].astype(np.float32)
            self.audio_paths = data["audio_paths"].astype(str)
        factor_ids = np.load(_resolve(self.root, catalog["factor_track_ids"]), allow_pickle=False).astype(str)
        if not np.array_equal(self.track_ids, factor_ids):
            raise ValueError("retrieval factor rows do not match the music catalog")
        self.factors = {
            name: np.load(_resolve(self.root, value["path"]), mmap_mode="r", allow_pickle=False)
            for name, value in catalog["factors"].items()
        }
        if any(array.shape != self.audio_embeddings.shape for array in self.factors.values()):
            raise ValueError("retrieval factor shape mismatch")
        with np.load(_resolve(self.root, catalog["metadata"]), allow_pickle=False) as data:
            ids = data["track_ids"].astype(str)
            artists = data["artist_ids"].astype(str)
            albums = data["album_ids"].astype(str)
            splits = data["splits"].astype(str)
            self.tag_labels = {}
            self.track_tags = {str(track_id): [] for track_id in ids}
            for factor in ("preference", "affect"):
                prompts = data[f"{factor}_prompts"].astype(str)
                targets = data[f"{factor}_targets"]
                for i, track_id in enumerate(ids):
                    self.track_tags[str(track_id)].extend(prompts[targets[i]].tolist())
        if len(set(ids.tolist())) != len(ids):
            raise ValueError("duplicate metadata track IDs")
        position = {track_id: index for index, track_id in enumerate(ids)}
        if set(position) != set(self.track_ids.tolist()):
            raise ValueError("catalog and metadata track IDs differ")
        take = np.asarray([position[track_id] for track_id in self.track_ids], dtype=np.int64)
        self.artist_ids = artists[take]
        self.album_ids = albums[take]
        self.catalog_splits = splits[take]
        self.catalog_position = {str(track_id): i for i, track_id in enumerate(self.track_ids)}

    def _load_affect_terms(self) -> None:
        path = _resolve(self.root, self.config["affect_direction"]["public_terms"])
        with np.load(path, allow_pickle=False) as data:
            terms = data["terms"].astype(str)
            values = data["term_va"].astype(np.float64)
        self.affect_terms = {term.lower(): value for term, value in zip(terms, values, strict=True)}

    def _public_affect_value(self, text: str) -> np.ndarray | None:
        normalized = " ".join(text.lower().strip().split())
        if normalized in self.affect_terms:
            return self.affect_terms[normalized]
        matched = [
            self.affect_terms[token]
            for token in re.findall(r"[a-z]+", normalized)
            if token in self.affect_terms
        ]
        return np.mean(matched, axis=0) if matched else None

    @torch.inference_mode()
    def _encode_queries(self, current: str, target: str, preference: str, method: str = "ours") -> tuple[np.ndarray, np.ndarray]:
        texts = [
            f"{current} {preference}".strip(),
            f"{target} {preference}".strip(),
            current,
            target,
            preference or "music",
        ]
        tokens = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=int(self.config["backbone"]["max_text_length"]),
        )
        raw = self.encoder.encode(
            tokens["input_ids"].to(self.device), tokens["attention_mask"].to(self.device)
        )
        semantic = self.retriever.encode_text("semantic", raw[:2])
        affect = self.retriever.encode_text("affect", raw[2:4])
        preference_value = self.retriever.encode_text("preference", raw[4:5])
        query_factors = torch.stack((semantic[0], semantic[1], affect[0], affect[1], preference_value[0]))
        predicted_va = self.text_va(affect)
        if method == "clamp3":
            query_factors = raw
        return query_factors.float().cpu().numpy(), predicted_va.float().cpu().numpy()

    def _scores(self, query_factors: np.ndarray, has_preference: bool = True, method: str = "ours") -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        semantic_start, semantic_target, affect_start, affect_target, preference = query_factors
        if method == "clamp3":
            return self.audio_embeddings @ semantic_start, self.audio_embeddings @ semantic_target, self.audio_embeddings @ preference
        weights = np.asarray(self.config["retriever"]["factor_weights"], dtype=np.float32)
        if not has_preference:
            weights[2] = 0
        weights /= weights.sum()
        preference_scores = np.asarray(self.factors["preference"] @ preference, dtype=np.float32)
        start_scores = (
            weights[0] * (self.factors["semantic"] @ semantic_start)
            + weights[1] * (self.factors["affect"] @ affect_start)
            + weights[2] * preference_scores
        ).astype(np.float32)
        target_scores = (
            weights[0] * (self.factors["semantic"] @ semantic_target)
            + weights[1] * (self.factors["affect"] @ affect_target)
            + weights[2] * preference_scores
        ).astype(np.float32)
        return start_scores, target_scores, preference_scores

    def _candidates(
        self,
        start_scores: np.ndarray,
        target_scores: np.ndarray,
        preference_scores: np.ndarray,
        allowed: np.ndarray | None = None,
        has_preference: bool = True,
    ) -> list[Candidate]:
        size = int(self.config["retriever"]["top_per_route"])
        allowed = np.ones(len(self.track_ids), dtype=bool) if allowed is None else allowed
        start_order = _stable_order(start_scores, self.track_ids)
        target_order = _stable_order(target_scores, self.track_ids)
        preference_order = _stable_order(preference_scores, self.track_ids)
        start_order = start_order[allowed[start_order]]
        target_order = target_order[allowed[target_order]]
        preference_order = preference_order[allowed[preference_order]]
        routes = [start_order[:size], target_order[:size]]
        if has_preference:
            routes.append(preference_order[:size])
        selected = sorted(set(np.concatenate(routes).tolist()))
        if len(selected) > 3 * size:
            raise AssertionError("candidate union exceeded the frozen budget")
        start_rank = np.empty(len(self.track_ids), dtype=np.int64)
        target_rank = np.empty(len(self.track_ids), dtype=np.int64)
        preference_rank = np.empty(len(self.track_ids), dtype=np.int64)
        start_rank[start_order] = np.arange(len(start_order))
        target_rank[target_order] = np.arange(len(target_order))
        preference_rank[preference_order] = np.arange(len(preference_order))
        denominator = max(int(allowed.sum()) - 1, 1)
        start_top = set(start_order[: int(self.config["endpoint"]["start_top"])].tolist())
        target_top = set(target_order[: int(self.config["endpoint"]["target_top"])].tolist())
        output = []
        for index in selected:
            embedding = self.audio_embeddings[index]
            embedding = embedding / max(float(np.linalg.norm(embedding)), 1.0e-12)
            output.append(Candidate(
                track_id=str(self.track_ids[index]),
                artist_id=str(self.artist_ids[index]),
                audio_path=str(self.audio_paths[index]),
                start_relevance=float(1.0 - start_rank[index] / denominator),
                target_relevance=float(1.0 - target_rank[index] / denominator),
                preference_percentile=float(1.0 - preference_rank[index] / denominator) if has_preference else 1.0,
                embedding=embedding.astype(np.float32),
                start_anchor=index in start_top,
                target_anchor=index in target_top,
            ))
        return output

    def recommend(self, current_affect: str, target_affect: str, music_preference: str = "", *, method: str = "ours", scope: str = "all_library") -> dict[str, Any]:
        current = str(current_affect).strip()
        target = str(target_affect).strip()
        preference = str(music_preference).strip()
        if method not in {"ours", "clamp3"} or scope not in {"all_library", "unseen_library"}:
            raise ValueError("Unknown method or library scope")
        if not current or not target:
            return {
                "status": "invalid_request",
                "failure_reason": "current_and_target_required",
                "tracks": [],
            }
        with self._lock:
            factors, predicted_va = self._encode_queries(current, target, preference, method)
            public_current = self._public_affect_value(current)
            public_target = self._public_affect_value(target)
            if public_current is not None and public_target is not None:
                affect_va = np.stack((public_current, public_target))
                direction_source = "public_affect_terms"
            else:
                affect_va = predicted_va
                direction_source = "emobank_text_va_fallback"
            direction = direction_from_va(
                affect_va[0], affect_va[1], float(self.config["affect_direction"]["tolerance"])
            )
            start, target_score, pref = self._scores(factors, bool(preference), method)
            allowed = self.catalog_splits == "test" if scope == "unseen_library" else np.ones(len(self.track_ids), dtype=bool)
            candidates = self._candidates(start, target_score, pref, allowed, bool(preference))
            if len(candidates) < 5:
                return {"status": "insufficient_catalog", "failure_reason": "candidate_pool_lt_5", "tracks": []}
            probabilities = self.relation.probabilities(np.stack([item.embedding for item in candidates]))
            edge = directional_scores(probabilities, direction, endpoint=False)
            endpoint = directional_scores(probabilities, direction, endpoint=True)
            search = self.config["search"]
            result = select_music_sequence(
                candidates,
                edge,
                endpoint,
                preference_threshold=float(search["preference_threshold"]),
                endpoint_threshold=float(search["endpoint_threshold"]),
                edge_threshold=float(search["edge_threshold"]),
                search_width=int(search["search_width"]),
                has_preference=bool(preference),
                maintain=all(value == 0 for value in direction),
            )
            chosen = result.indices
            if method == "clamp3":
                # Original CLaMP3 baseline: five highest cosine scores for the
                # target-plus-preference text, with no trained reranking.
                chosen = tuple(sorted(range(len(candidates)), key=lambda i: (-candidates[i].target_relevance, candidates[i].track_id))[:5])
            if result.status not in {"transition_found", "maintain_found"} or len(chosen) != 5:
                if method == "ours":
                    chosen = fallback_sequence(candidates, bool(preference))
            final_preference = float(np.mean([
                candidates[index].preference_percentile for index in chosen
            ]))
            tracks = [
                {
                    "position": position + 1,
                    "track_id": candidates[index].track_id,
                    "artist_id": candidates[index].artist_id,
                    "audio_path": candidates[index].audio_path,
                    "preference_percentile": candidates[index].preference_percentile,
                }
                for position, index in enumerate(chosen)
            ]
            if len({track["track_id"] for track in tracks}) != 5:
                raise AssertionError("the final sequence is not unique")
            return {
                "status": result.status if method == "ours" else "baseline_retrieved",
                "failure_reason": result.failure_reason if method == "ours" else None,
                "method": method,
                "scope": scope,
                "request": {
                    "current_affect": current,
                    "target_affect": target,
                    "music_preference": preference,
                },
                "direction": {"valence": direction[0], "arousal": direction[1]},
                "direction_source": direction_source,
                "request_va": affect_va.tolist(),
                "candidate_count": len(candidates),
                "library_count": int(allowed.sum()),
                "metrics": {
                    "attempted_mean_edge": result.mean_edge,
                    "attempted_weakest_edge": result.weakest_edge,
                    "attempted_endpoint_score": result.endpoint_score,
                    "final_mean_preference_percentile": final_preference,
                    "final_mean_edge": float(np.mean([edge[left, right] for left, right in zip(chosen[:-1], chosen[1:])])),
                    "final_weakest_edge": float(min(edge[left, right] for left, right in zip(chosen[:-1], chosen[1:]))),
                    "final_endpoint_score": float(endpoint[chosen[0], chosen[-1]]),
                },
                "tracks": tracks,
            }
