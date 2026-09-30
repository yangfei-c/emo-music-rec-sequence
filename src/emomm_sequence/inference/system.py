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

from .candidate import Candidate
from ..encoding.text import FrozenTextEncoder
from ..models.frozen_retriever import FrozenRetriever
from ..models.text_va import load_text_va
from ..calibration.cqr import VAUncertaintyModel


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
    """Load frozen assets once and return ordered music sequences."""

    def __init__(
        self,
        root: str | Path | None = None,
        config: str | Path = "configs/system_route2.yaml",
        device: str | None = None,
        verify_hashes: bool = True,
    ) -> None:
        self.root = Path(root) if root is not None else Path(__file__).resolve().parents[3]
        config_path = _resolve(self.root, config)
        self.config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        if "route2" not in self.config:
            raise ValueError("Only Route2 configurations are supported")
        self.route2 = True
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
        route = self.config["route2"]
        self.va_model = VAUncertaintyModel.from_artifacts(
            _resolve(self.root, route["checkpoint"]),
            _resolve(self.root, route["calibration"]),
            device=str(self.device),
        )
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
        route = self.config["route2"]
        self._verify(route["checkpoint"], route["checkpoint_sha256"])
        self._verify(route["calibration"], route["calibration_sha256"])

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

    def _recommend_route2(
        self,
        current: str,
        target: str,
        preference: str,
        *,
        method: str,
        scope: str,
        num_tracks: int,
    ) -> dict[str, Any]:
        """Plan on calibrated continuous music VA, with an explicit sequence length."""
        from .search import rank_relevance, select_robust_sequence

        route = self.config["route2"]
        if num_tracks < 1 or num_tracks > int(route["max_tracks"]):
            raise ValueError(f"歌曲首数应在 1 到 {route['max_tracks']} 之间")
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
            direction = tuple(int(np.sign(value)) for value in affect_va[1] - affect_va[0])
            start_scores, target_scores, preference_scores = self._scores(factors, bool(preference), method)
            allowed = self.catalog_splits == "test" if scope == "unseen_library" else np.ones(len(self.track_ids), dtype=bool)
            candidates = self._candidates(start_scores, target_scores, preference_scores, allowed, bool(preference))
            if len(candidates) < num_tracks:
                return {
                    "status": "insufficient_catalog", "failure_reason": "candidate_pool_lt_requested_length",
                    "tracks": [], "method": method, "scope": scope, "requested_num_tracks": num_tracks,
                    "candidate_count": len(candidates), "library_count": int(allowed.sum()),
                }
            median, lower, upper = self.va_model.predict(np.stack([item.embedding for item in candidates]), source=None)
            ids = np.asarray([item.track_id for item in candidates])
            start_rank = rank_relevance(np.asarray([item.start_relevance for item in candidates]), ids)
            target_rank = rank_relevance(np.asarray([item.target_relevance for item in candidates]), ids)
            anchor_conflict = False
            search_result = None
            if method == "clamp3":
                chosen = tuple(sorted(range(len(candidates)), key=lambda i: (-candidates[i].target_relevance, candidates[i].track_id))[:num_tracks])
                status, failure_reason = "baseline_retrieved", None
            elif num_tracks == 1:
                target_index = int(np.argmax(target_rank))
                search_result = select_robust_sequence(
                    median, lower, upper, start_rank, target_rank,
                    num_tracks=1, start_index=target_index, target_index=target_index,
                    direction=direction, lambda_retrieval=float(route["lambda_retrieval"]),
                    beam_width=int(route["beam_width"]), track_ids=ids,
                    mode="robust" if method == "ours" else "point",
                )
                chosen = search_result.indices
                status, failure_reason = search_result.status, search_result.failure_reason
            else:
                pairs = []
                for s, first in enumerate(candidates):
                    if not first.start_anchor:
                        continue
                    for t, last in enumerate(candidates):
                        if s == t or not last.target_anchor:
                            continue
                        compatible = all(direction[axis] == 0 or direction[axis] * (median[t, axis] - median[s, axis]) >= 0 for axis in range(2))
                        if compatible:
                            pairs.append((-(start_rank[s] + target_rank[t]), first.track_id, last.track_id, s, t))
                if not pairs:
                    chosen = tuple(sorted(range(len(candidates)), key=lambda i: (-candidates[i].target_relevance, candidates[i].track_id))[:num_tracks])
                    status, failure_reason = "anchor_direction_conflict", "no_direction_compatible_anchor_pair"
                    anchor_conflict = True
                else:
                    _, _, _, start_index, target_index = min(pairs)
                    search_result = select_robust_sequence(
                        median, lower, upper, start_rank, target_rank,
                        num_tracks=num_tracks, start_index=start_index, target_index=target_index,
                        direction=direction, lambda_retrieval=float(route["lambda_retrieval"]),
                        beam_width=int(route["beam_width"]), track_ids=ids,
                        mode="robust" if method == "ours" else "point",
                    )
                    chosen = tuple(search_result.indices)
                    status, failure_reason = search_result.status, search_result.failure_reason
            if len(chosen) != num_tracks or len(set(chosen)) != num_tracks:
                raise AssertionError("route 2 must return exactly the requested number of unique tracks")
            tracks = [
                {
                    "position": position + 1,
                    "track_id": candidates[index].track_id,
                    "artist_id": candidates[index].artist_id,
                    "audio_path": candidates[index].audio_path,
                    "preference_percentile": candidates[index].preference_percentile,
                    "music_v": float(median[index, 0]),
                    "music_a": float(median[index, 1]),
                    "music_va_lower": lower[index].tolist(),
                    "music_va_upper": upper[index].tolist(),
                }
                for position, index in enumerate(chosen)
            ]
            metrics = {
                "robust_path_cost": None if search_result is None else search_result.robust_path_cost,
                "point_path_cost": None if search_result is None else search_result.point_path_cost,
                "retrieval_match_cost": None if search_result is None else search_result.retrieval_match_cost,
                "objective": None if search_result is None else search_result.objective,
                "reference_step": None if search_result is None else list(search_result.reference_step),
                "final_mean_preference_percentile": float(np.mean([candidates[i].preference_percentile for i in chosen])),
            }
            return {
                "status": status, "failure_reason": failure_reason,
                "method": method, "scope": scope, "requested_num_tracks": num_tracks,
                "request": {"current_affect": current, "target_affect": target, "music_preference": preference},
                "direction": {"valence": direction[0], "arousal": direction[1]},
                "direction_source": direction_source,
                "request_va": affect_va.tolist(),
                "candidate_count": len(candidates), "library_count": int(allowed.sum()),
                "calibration_scope": "DEAM/PMEmo song-level marginal; transfer to this catalog is unverified",
                "anchor_direction_conflict": anchor_conflict,
                "metrics": metrics, "tracks": tracks,
            }

    def recommend(self, current_affect: str, target_affect: str, music_preference: str = "", *, method: str = "ours", scope: str = "all_library", num_tracks: int = 5) -> dict[str, Any]:
        current = str(current_affect).strip()
        target = str(target_affect).strip()
        preference = str(music_preference).strip()
        if method not in {"ours", "point", "clamp3"} or scope not in {"all_library", "unseen_library"}:
            raise ValueError("Unknown method or library scope")
        if not current or not target:
            return {
                "status": "invalid_request",
                "failure_reason": "current_and_target_required",
                "tracks": [],
            }
        return self._recommend_route2(current, target, preference, method=method, scope=scope, num_tracks=num_tracks)
