"""Parse one natural-language request into the three frozen experiment slots.

This parser is a demo adapter. It stays separate from retrieval and sequence
selection so formal experiments can consume frozen structured queries without
introducing an LLM as an uncontrolled experimental variable.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from typing import Any


class QueryParsingError(ValueError):
    """Raised when a request cannot be mapped to all required fields."""


@dataclass(frozen=True)
class StructuredMusicRequest:
    raw_text: str
    current_affect: str
    target_affect: str
    music_preference: str
    evidence: dict[str, str]
    parser_backend: str
    parser_model: str
    parser_model_digest: str
    defaulted_fields: tuple[str, ...] = ()
    num_tracks: int | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_retrieval_queries(request: StructuredMusicRequest) -> dict[str, str]:
    """Build start, target, and preference queries from a parsed request."""
    return {
        "start": f"{request.current_affect} {request.music_preference}".strip(),
        "target": f"{request.target_affect} {request.music_preference}".strip(),
        "preference": request.music_preference.strip(),
    }


OUTPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "current_affect": {"type": "string", "description": "Current feeling, concise English, or empty if absent"},
        "target_affect": {"type": "string", "description": "Desired feeling, concise English, or empty if absent"},
        "music_preference": {"type": "string", "description": "Music style or instrument in English; preserve exclusions; empty if absent"},
        "num_tracks": {"type": "integer", "description": "Requested number of songs; 0 if absent"},
        "current_affect_evidence": {"type": "string"},
        "target_affect_evidence": {"type": "string"},
        "music_preference_evidence": {"type": "string"},
        "num_tracks_evidence": {"type": "string"},
        "missing_fields": {
            "type": "array",
            "items": {
                "type": "string",
                "enum": ["current_affect", "target_affect", "music_preference"],
            },
        },
    },
    "required": [
        "current_affect",
        "target_affect",
        "music_preference",
        "num_tracks",
        "current_affect_evidence",
        "target_affect_evidence",
        "music_preference_evidence",
        "num_tracks_evidence",
        "missing_fields",
    ],
    "additionalProperties": False,
}


SYSTEM_PROMPT = """你是音乐需求信息抽取器。只输出 JSON，不回答用户，也不给建议。
从用户的一句话中提取：current_affect（当前心情）、target_affect（希望达到的心情）、music_preference（音乐类型或乐器偏好）、num_tracks（明确要求推荐的歌曲首数）。
三个值必须翻译成简短、准确的英文检索短语。“烦躁”是 annoyed 或 irritated，不是 frightened；frightened 只表示恐惧。
三个 evidence 字段必须逐字复制用户句子里的对应中文或英文片段，不能用英文翻译替代原文证据。
用户没说出的项目输出空字符串，对应 evidence 也为空，并把字段名加入 missing_fields。
用户说“保持”时，目标情绪等于明确说出的当前情绪。禁止把 evidence 等字段名填作内容。
保留偏好中的否定，例如“不想听摇滚”应表达为 no rock，不能把摇滚当作喜欢的类型。
偏好只能包含本次用户原文明确提到的要求。不得从示例、常识或情绪推断人声限制、乐器、风格；只说“爵士”就输出 jazz，不能自动添加 instrumental 或 no vocals。示例中的限制不适用于其他输入。
阅读全文后再填写 music_preference，把明确喜欢的风格、乐器和明确排除的内容合并。句末的“不想听/不要/不带”等排除要求也必须保留；不能提取了前面的喜欢项就漏掉后面的排除项。music_preference_evidence 应覆盖包含这些要求的原文片段。输出前核对：没有凭空增加限制，也没有遗漏原文的限制。
只有明确说出“几首/几首歌”等数量时才填 num_tracks，否则填 0。num_tracks_evidence 必须逐字引用原文中的数量片段；未提数量时为空。不能把日期、年龄等数字当作歌曲首数。

示例输入：心里有点焦虑，希望放松一下，来 7 首轻柔的钢琴曲。
示例输出：{"current_affect":"anxious","target_affect":"relaxed","music_preference":"soft piano music","num_tracks":7,"current_affect_evidence":"有点焦虑","target_affect_evidence":"放松一下","music_preference_evidence":"轻柔的钢琴","num_tracks_evidence":"7 首","missing_fields":[]}

示例输入：有些失落。
示例输出：{"current_affect":"sad","target_affect":"","music_preference":"","num_tracks":0,"current_affect_evidence":"有些失落","target_affect_evidence":"","music_preference_evidence":"","num_tracks_evidence":"","missing_fields":["target_affect","music_preference"]}

示例输入：我很开心，继续保持，想听爵士。
示例输出：{"current_affect":"happy","target_affect":"happy","music_preference":"jazz","num_tracks":0,"current_affect_evidence":"很开心","target_affect_evidence":"继续保持","music_preference_evidence":"爵士","num_tracks_evidence":"","missing_fields":[]}
"""


class NaturalLanguageQueryParser:
    """Frozen Qwen3:4B parser with JSON-schema-constrained output."""

    def __init__(
        self,
        model: str = "qwen3:4b",
        base_url: str = "http://127.0.0.1:11434",
        expected_digest: str = "",
        strict_digest: bool = True,
        timeout_seconds: float = 120.0,
        seed: int = 42,
    ) -> None:
        self.model = str(model)
        self.base_url = base_url.rstrip("/")
        self.expected_digest = str(expected_digest)
        self.strict_digest = bool(strict_digest)
        self.timeout_seconds = float(timeout_seconds)
        self.seed = int(seed)

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "NaturalLanguageQueryParser":
        if str(config.get("backend", "ollama")) != "ollama":
            raise QueryParsingError("only the frozen Ollama parser backend is currently supported")
        return cls(
            model=config["model"],
            base_url=config.get("base_url", "http://127.0.0.1:11434"),
            expected_digest=config.get("expected_digest", ""),
            strict_digest=config.get("strict_digest", True),
            timeout_seconds=config.get("timeout_seconds", 120.0),
            seed=config.get("seed", 42),
        )

    def _request(self, path: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        data = None if payload is None else json.dumps(payload, ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(
            f"{self.base_url}{path}",
            data=data,
            headers={"Content-Type": "application/json"},
            method="GET" if payload is None else "POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                return json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as error:
            raise QueryParsingError(f"frozen query parser is unavailable: {error}") from error

    def model_digest(self) -> str:
        response = self._request("/api/tags")
        for record in response.get("models", []):
            if record.get("name") == self.model or record.get("model") == self.model:
                digest = str(record.get("digest", ""))
                if self.expected_digest and self.strict_digest and digest != self.expected_digest:
                    raise QueryParsingError(
                        f"parser model digest mismatch for {self.model}: {digest or 'missing'}"
                    )
                return digest
        raise QueryParsingError(f"frozen parser model is not installed: {self.model}")

    def parse(self, text: str) -> StructuredMusicRequest:
        raw_text = str(text).strip()
        if not raw_text:
            raise QueryParsingError("request_text is required")
        if len(raw_text) > 1000:
            raise QueryParsingError("输入请控制在 1000 字以内")
        digest = self.model_digest()
        payload = {
            "model": self.model,
            "stream": False,
            "think": False,
            "keep_alive": "2m",
            "format": OUTPUT_SCHEMA,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": raw_text},
            ],
            "options": {"temperature": 0, "seed": self.seed, "num_ctx": 4096, "num_predict": 512},
        }
        response = self._request("/api/chat", payload)
        try:
            content = response["message"]["content"]
            parsed = json.loads(content) if isinstance(content, str) else dict(content)
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            raise QueryParsingError("parser returned invalid structured output") from error

        required = ("current_affect", "target_affect", "music_preference")
        values = {name: str(parsed.get(name, "")).strip() for name in required}
        missing = {str(name) for name in parsed.get("missing_fields", [])}
        missing.update(name for name, value in values.items() if not value)
        if "current_affect" in missing:
            raise QueryParsingError("还没有识别到当前心情，请补充一句，例如：我现在很烦。")
        evidence = {
            "current_affect": str(parsed.get("current_affect_evidence", "")).strip(),
            "target_affect": str(parsed.get("target_affect_evidence", "")).strip(),
            "music_preference": str(parsed.get("music_preference_evidence", "")).strip(),
        }
        for field, span in evidence.items():
            if span and span not in raw_text:
                raise QueryParsingError("情绪解析的原文依据校验失败，请换一种表述后重试。")
            if values[field] and not span:
                raise QueryParsingError("情绪解析缺少原文依据，请换一种表述后重试。")
            if values[field].lower() in {"evidence", "current_affect", "target_affect", "music_preference"}:
                raise QueryParsingError("情绪解析返回无效内容，请重试。")
        raw_number = parsed.get("num_tracks", 0)
        if isinstance(raw_number, bool) or not isinstance(raw_number, int):
            raise QueryParsingError("歌曲首数解析无效，请明确写出需要几首歌。")
        parsed_num_tracks = raw_number or None
        number_evidence = str(parsed.get("num_tracks_evidence", "")).strip()
        if parsed_num_tracks is not None and (not number_evidence or number_evidence not in raw_text):
            raise QueryParsingError("歌曲首数缺少原文依据，请明确写出需要几首歌。")
        if parsed_num_tracks is None and number_evidence:
            raise QueryParsingError("歌曲首数和原文依据不一致。")
        evidence["num_tracks"] = number_evidence
        # Missing preference is not invented. A missing goal is an explicit,
        # disclosed product default, not a claim about the user's actual wish.
        defaulted = []
        if "target_affect" in missing:
            values["target_affect"] = "calm"
            evidence["target_affect"] = ""
            defaulted.append("target_affect")
        if "music_preference" in missing:
            values["music_preference"] = ""
            evidence["music_preference"] = ""
        return StructuredMusicRequest(
            raw_text=raw_text,
            current_affect=values["current_affect"],
            target_affect=values["target_affect"],
            music_preference=values["music_preference"],
            evidence=evidence,
            parser_backend="ollama",
            parser_model=self.model,
            parser_model_digest=digest,
            defaulted_fields=tuple(defaulted),
            num_tracks=parsed_num_tracks,
        )
