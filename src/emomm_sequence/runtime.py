"""One-utterance API shared by CLI and the second EmoMM-Web page."""
from __future__ import annotations
from pathlib import Path

import numpy as np

from .query import NaturalLanguageQueryParser
from .system import EmoMMSequence


class EmoMMSequenceRuntime:
    def __init__(self, root: str | Path | None = None, config_path: str | Path = "configs/system.yaml", device: str | None = None):
        self.system = EmoMMSequence(root=root, config=config_path, device=device)
        self.parser = NaturalLanguageQueryParser.from_config(self.system.config.get("query_parser", {"model": "qwen3:4b"}))
        with np.load(self.system.root / self.system.config["catalog"]["music"], allow_pickle=False) as data:
            self.music_va = data["music_va"].astype(np.float32)

    def audio_path(self, track_id: str) -> Path:
        # IDs are looked up, never interpolated into a filesystem path.
        position = self.system.catalog_position[track_id]
        path = Path(self.system.audio_paths[position])
        if not path.is_file():
            raise FileNotFoundError(path)
        return path

    def recommend_from_text(self, request_text: str, method: str = "ours", scope: str = "unseen_library") -> dict:
        if method not in {"ours", "clamp3"}:
            raise ValueError("请选择 Ours 或 CLaMP3")
        if scope not in {"unseen_library", "all_library"}:
            raise ValueError("请选择本次训练留出曲库或完整曲库")
        parsed = self.parser.parse(request_text)
        result = self.system.recommend(parsed.current_affect, parsed.target_affect, parsed.music_preference, method=method, scope=scope)
        if len(result.get("tracks", [])) != 5:
            raise ValueError("当前没有足够的可用歌曲，请检查曲库")
        warnings = []
        if "target_affect" in parsed.defaulted_fields:
            warnings.append("你没有指定目标心情，本次暂按“平静”推荐。可以在输入中更改目标。")
        if not parsed.music_preference:
            warnings.append("未指定音乐偏好，本次不限制曲风或乐器。")
        if result["failure_reason"]:
            warnings.append("未找到满足情绪过渡约束的五首序列，当前返回偏好／目标相关的备选歌曲，不将它标为过渡成功。")
        warnings.append("VA 表示模型预测的歌曲情绪特征，不代表听者实际情绪变化。")
        tracks = []
        for item in result["tracks"]:
            track_id = item["track_id"]
            coordinate = self.music_va[self.system.catalog_position[track_id]]
            tracks.append({"position": item["position"], "track_id": track_id, "artist_id": item["artist_id"], "title": track_id,
                           "audio_url": f"/api/media/sequence/{track_id}", "tags": self.system.track_tags[track_id],
                           "music_v": float(coordinate[0]), "music_a": float(coordinate[1]),
                           "preference_relevance": item["preference_percentile"] if parsed.music_preference else None})
        return {"method": method, "model_label": "Ours" if method == "ours" else "CLaMP3", "model_version": self.system.config.get("run_id", "unversioned"),
                "text_va_checkpoint_sha256": self.system.config["text_va"]["checkpoint_sha256"],
                "music_va_checkpoints_sha256": self.system.config["relation"]["checkpoints_sha256"],
                "system_config_sha256": self.system.config_sha256,
                "retriever_checkpoint_sha256": self.system.config["retriever"]["checkpoint_sha256"] if method == "ours" else self.system.config["backbone"]["checkpoint_sha256"],
                "status": result["status"], "failure_reason": result["failure_reason"], "scope": scope, "input": parsed.as_dict(),
                "diagnostics": {"candidate_count": result["candidate_count"], "library_count": result["library_count"], "direction": result["direction"], "direction_source": result["direction_source"], "request_va": result["request_va"], "request_va_scale": [-1, 1], "music_va_scale": [0, 1], "retrieval_route": "semantic_preservation_mapped_queries_multiview"},
                "metrics": result["metrics"], "tracks": tracks, "warnings": warnings,
                "baseline_protocol": "目标与偏好文本的原始 CLaMP3 余弦相似度 Top-5；VA 仅用于同尺度展示。" if method == "clamp3" else None}
