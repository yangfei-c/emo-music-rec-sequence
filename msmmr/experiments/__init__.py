"""可复用的基础实验入口。"""

from msmmr.experiments.music import run_music_tagging, run_music_emotion
from msmmr.experiments.text import run_text_emotion

EXPERIMENTS = {
    "music_mtg_full183": lambda options: run_music_tagging("full183", options),
    "music_mtg_genre87": lambda options: run_music_tagging("genre87", options),
    "music_mtg_instrument40": lambda options: run_music_tagging(
        "instrument40", options
    ),
    "music_mtg_mood56": lambda options: run_music_tagging("mood56", options),
    "music_va_deam": lambda options: run_music_emotion("deam", options),
    "music_va_pmemo": lambda options: run_music_emotion("pmemo", options),
    "music_va_joint": lambda options: run_music_emotion("joint", options),
    "text_va": run_text_emotion,
}


def run_named_experiment(name: str, options):
    if name not in EXPERIMENTS:
        raise ValueError(f"未知实验：{name}；可选项：{', '.join(EXPERIMENTS)}")
    return EXPERIMENTS[name](options)


__all__ = [
    "EXPERIMENTS",
    "run_music_tagging",
    "run_music_emotion",
    "run_named_experiment",
    "run_text_emotion",
]
