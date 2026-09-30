"""One-utterance API shared by CLI and the second EmoMM-Web page."""
from __future__ import annotations
from pathlib import Path

from .query import NaturalLanguageQueryParser
from .system import EmoMMSequence


class EmoMMSequenceRuntime:
    def __init__(self, root: str | Path | None = None, config_path: str | Path = "configs/system_route2.yaml", device: str | None = None):
        self.system = EmoMMSequence(root=root, config=config_path, device=device)
        self.parser = NaturalLanguageQueryParser.from_config(self.system.config.get("query_parser", {"model": "qwen3:4b"}))

    def audio_path(self, track_id: str) -> Path:
        # IDs are looked up, never interpolated into a filesystem path.
        position = self.system.catalog_position[track_id]
        path = Path(self.system.audio_paths[position])
        if not path.is_file():
            raise FileNotFoundError(path)
        return path

    def recommend_from_text(self, request_text: str, method: str = "ours", scope: str = "unseen_library", num_tracks: int | None = None) -> dict:
        if method not in {"ours", "point", "clamp3"}:
            raise ValueError("请选择可用的序列方法")
        if scope not in {"unseen_library", "all_library"}:
            raise ValueError("请选择本次训练留出曲库或完整曲库")
        parsed = self.parser.parse(request_text)
        if num_tracks is not None and (isinstance(num_tracks, bool) or not isinstance(num_tracks, int)):
            raise ValueError("歌曲首数必须是整数")
        chosen_count = num_tracks if num_tracks is not None else parsed.num_tracks
        count_source = "explicit" if num_tracks is not None else "request_text" if parsed.num_tracks is not None else "default"
        if chosen_count is None:
            chosen_count = int(self.system.config["route2"].get("default_tracks", 5))
        result = self.system.recommend(parsed.current_affect, parsed.target_affect, parsed.music_preference, method=method, scope=scope, num_tracks=chosen_count)
        if len(result.get("tracks", [])) != chosen_count:
            raise ValueError("当前没有足够的可用歌曲，请检查曲库")
        warnings = []
        if num_tracks is not None and parsed.num_tracks is not None and num_tracks != parsed.num_tracks:
            warnings.append(f"文字中要求 {parsed.num_tracks} 首，界面明确指定 {num_tracks} 首；本次按界面首数执行。")
        if count_source == "default":
            warnings.append(f"未指定歌曲首数，本次默认推荐 {chosen_count} 首。")
        if "target_affect" in parsed.defaulted_fields:
            warnings.append("你没有指定目标心情，本次暂按“平静”推荐。可以在输入中更改目标。")
        if not parsed.music_preference:
            warnings.append("未指定音乐偏好，本次不限制曲风或乐器。")
        if result["failure_reason"]:
            warnings.append(f"未找到满足方向要求的 {chosen_count} 首序列；当前是目标相关备选，不将它标为过渡成功。")
        warnings.append("VA 表示模型预测的歌曲情绪特征，不代表听者实际情绪变化。")
        warnings.append("音乐 VA 区间按 DEAM/PMEmo 校准；对当前曲库中被筛选的歌曲不提供 90% 覆盖保证。")
        tracks = []
        for item in result["tracks"]:
            track_id = item["track_id"]
            tracks.append({"position": item["position"], "track_id": track_id, "artist_id": item["artist_id"], "title": track_id,
                           "audio_url": f"/api/media/sequence/{track_id}", "tags": self.system.track_tags[track_id],
                           "music_v": float(item["music_v"]), "music_a": float(item["music_a"]),
                           "music_va_lower": item.get("music_va_lower"), "music_va_upper": item.get("music_va_upper"),
                           "preference_relevance": item["preference_percentile"] if parsed.music_preference else None})
        return {"method": method, "model_label": {"ours": "CQR + 稳健序列", "point": "点估计序列", "clamp3": "CLaMP3"}[method], "model_version": self.system.config.get("run_id", "unversioned"),
                "text_va_checkpoint_sha256": self.system.config["text_va"]["checkpoint_sha256"],
                "music_va_checkpoints_sha256": {"quantile": self.system.config["route2"]["checkpoint_sha256"]},
                "system_config_sha256": self.system.config_sha256,
                "retriever_checkpoint_sha256": self.system.config["retriever"]["checkpoint_sha256"] if method == "ours" else self.system.config["backbone"]["checkpoint_sha256"],
                "status": result["status"], "failure_reason": result["failure_reason"], "scope": scope, "input": {**parsed.as_dict(), "num_tracks": chosen_count, "num_tracks_source": count_source},
                "diagnostics": {"candidate_count": result["candidate_count"], "library_count": result["library_count"], "direction": result["direction"], "direction_source": result["direction_source"], "request_va": result["request_va"], "request_va_scale": [-1, 1], "music_va_scale": [0, 1], "retrieval_route": "semantic_preservation_mapped_queries_multiview", "calibration_scope": result.get("calibration_scope")},
                "metrics": result["metrics"], "tracks": tracks, "warnings": warnings,
                "baseline_protocol": "目标与偏好文本的原始 CLaMP3 余弦相似度 Top-N；VA 仅用于同尺度展示。" if method == "clamp3" else None}
