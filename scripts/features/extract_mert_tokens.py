"""Extract CLaMP3-compatible 5-second MERT tokens with batched chunks."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import torch
from huggingface_hub import snapshot_download
from transformers import Wav2Vec2FeatureExtractor


ROOT = Path(__file__).resolve().parents[2]
CLAMP_AUDIO = ROOT / "third_party" / "CLaMP3" / "preprocessing" / "audio"
if str(CLAMP_AUDIO) not in sys.path:
    sys.path.insert(0, str(CLAMP_AUDIO))

from MusicHubert import MusicHubertModel
from configuration_musichubert import MusicHubertConfig

SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from emomm_sequence.data.audio import select_front_middle_end


MODEL_ID = "m-a-p/MERT-v1-95M"
MODEL_REVISION = "12af15fef9d0ac838c3f475bfbbf26d2060dd4f5"
SAMPLE_RATE = 24000
WINDOW_SECONDS = 5
MIN_SECONDS = 1


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_frozen_mert(model_root: str | Path, device: torch.device):
    """Strictly load the pinned checkpoint across weight-norm API versions.

    New Transformers converts legacy weight_g/v names during from_pretrained;
    that conversion is incompatible with this vendored MusicHubert class.
    Map by the actual module state keys, so no missing key is randomized.
    """
    model_config = MusicHubertConfig.from_pretrained(model_root)
    if not hasattr(model_config, "conv_pos_batch_norm"):
        model_config.conv_pos_batch_norm = False
    model = MusicHubertModel(model_config)
    state = torch.load(Path(model_root) / "pytorch_model.bin", map_location="cpu", weights_only=True)
    expected = model.state_dict()
    for old, new in (("weight_g", "parametrizations.weight.original0"),
                     ("weight_v", "parametrizations.weight.original1")):
        old_key = "encoder.pos_conv_embed.conv." + old
        new_key = "encoder.pos_conv_embed.conv." + new
        if old_key in state and new_key in expected:
            state[new_key] = state.pop(old_key)
    model.load_state_dict(state, strict=True)
    return model.to(device).eval().requires_grad_(False)


def chunks(waveform: np.ndarray) -> list[np.ndarray]:
    width = SAMPLE_RATE * WINDOW_SECONDS
    values = [waveform[start : start + width] for start in range(0, len(waveform), width)]
    if values and len(values[-1]) < SAMPLE_RATE * MIN_SECONDS:
        values.pop()
    return values


def select_track_windows(
    waveform: np.ndarray, policy: str, window_seconds: int
) -> np.ndarray:
    """Select a reproducible full-track view before 5-second tokenization."""

    if policy == "full":
        return waveform
    if policy != "front_middle_end":
        raise ValueError(f"unsupported window policy: {policy}")
    return select_front_middle_end(waveform, SAMPLE_RATE, window_seconds)


@torch.inference_mode()
def encode_equal_length(
    values: list[np.ndarray],
    processor: Wav2Vec2FeatureExtractor,
    model: MusicHubertModel,
    device: torch.device,
) -> torch.Tensor:
    processed = processor(
        values,
        return_tensors="pt",
        sampling_rate=SAMPLE_RATE,
        padding=True,
        do_normalize=True,
    ).input_values.to(device)
    hidden = model(processed, output_hidden_states=True).hidden_states
    return torch.stack([layer.mean(dim=1) for layer in hidden]).mean(dim=0).float().cpu()


def encode_file(
    path: Path,
    processor: Wav2Vec2FeatureExtractor,
    model: MusicHubertModel,
    device: torch.device,
    window_policy: str,
    window_seconds: int,
) -> np.ndarray:
    waveform, _ = librosa.load(path, sr=SAMPLE_RATE, mono=True)
    waveform = select_track_windows(waveform, window_policy, window_seconds)
    values = chunks(waveform.astype(np.float32, copy=False))
    if not values:
        raise ValueError(f"audio shorter than {MIN_SECONDS}s: {path}")
    width = SAMPLE_RATE * WINDOW_SECONDS
    full = [value for value in values if len(value) == width]
    remainder = [value for value in values if len(value) != width]
    encoded = []
    if full:
        encoded.append(encode_equal_length(full, processor, model, device))
    for value in remainder:
        encoded.append(encode_equal_length([value], processor, model, device))
    result = torch.cat(encoded, dim=0).unsqueeze(0).numpy().astype(np.float32)
    if result.ndim != 3 or result.shape[0] != 1 or result.shape[2] != 768:
        raise ValueError(f"invalid MERT output {result.shape}: {path}")
    if not np.isfinite(result).all():
        raise ValueError(f"MERT output has NaN/Inf: {path}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--split", help="exact manifest split value to include")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--window-policy",
        choices=("full", "front_middle_end"),
        default="full",
    )
    parser.add_argument("--window-seconds", type=int, default=30)
    args = parser.parse_args()
    if args.manifest:
        manifest = args.manifest.resolve()
        frame = pd.read_csv(manifest, keep_default_na=False)
        if args.split:
            frame = frame[frame["split"].astype(str) == args.split]
        jobs = [
            (Path(row.audio_path).resolve(), Path(row.feature_path).resolve())
            for row in frame.itertuples(index=False)
        ]
        input_root = manifest
        output_root = (
            args.output.resolve()
            if args.output
            else (jobs[0][1].parent if jobs else ROOT / "artifacts/features/mert")
        )
    else:
        if args.input is None or args.output is None:
            parser.error("--input and --output are required unless --manifest is used")
        input_root = args.input.resolve()
        output_root = args.output.resolve()
        files = sorted(
            path
            for path in input_root.rglob("*")
            if path.suffix.lower() in {".mp3", ".m4a", ".wav", ".flac", ".ogg"}
            and not path.name.startswith("._")
            and "__MACOSX" not in path.parts
        )
        jobs = [
            ((path), (output_root / path.relative_to(input_root)).with_suffix(".npy"))
            for path in files
        ]
        if input_root.is_file():
            if input_root.suffix.lower() not in {".mp3", ".m4a", ".wav", ".flac", ".ogg"}:
                parser.error("--input file must be an audio file")
            destination = output_root if output_root.suffix.lower() == ".npy" else output_root / (input_root.stem + ".npy")
            jobs = [(input_root, destination)]
            output_root = destination.parent
    if not jobs:
        raise FileNotFoundError(f"no audio under {input_root}")
    model_root = snapshot_download(repo_id=MODEL_ID, revision=MODEL_REVISION)
    processor = Wav2Vec2FeatureExtractor(
        feature_size=1,
        sampling_rate=SAMPLE_RATE,
        padding_value=0.0,
        return_attention_mask=True,
        do_normalize=True,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_frozen_mert(model_root, device)
    output_root.mkdir(parents=True, exist_ok=True)
    started, processed, skipped, failures = time.time(), 0, 0, []
    for position, (path, output) in enumerate(jobs, start=1):
        if output.is_file() and not args.overwrite:
            skipped += 1
        else:
            output.parent.mkdir(parents=True, exist_ok=True)
            try:
                np.save(
                    output,
                    encode_file(
                        path,
                        processor,
                        model,
                        device,
                        args.window_policy,
                        args.window_seconds,
                    ),
                )
                processed += 1
            except Exception as error:
                failures.append({"path": path.as_posix(), "error": repr(error)})
        if position % 25 == 0 or position == len(jobs):
            elapsed = max(time.time() - started, 1.0e-9)
            rate = position / elapsed
            eta = (len(jobs) - position) / max(rate, 1.0e-9)
            print(
                f"MERT {position}/{len(jobs)} | processed={processed} skipped={skipped} "
                f"failed={len(failures)} | {rate:.2f} tracks/s | ETA={eta/60:.1f} min",
                flush=True,
            )
    report = {
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "input": input_root.as_posix(),
        "output": output_root.as_posix(),
        "audio_files": len(jobs),
        "processed": processed,
        "skipped": skipped,
        "failures": failures,
        "feature_files": sum(output.is_file() for _, output in jobs),
        "window_seconds": WINDOW_SECONDS,
        "track_window_policy": args.window_policy,
        "track_window_seconds": args.window_seconds,
        "sample_rate": SAMPLE_RATE,
        "device": str(device),
        "manifest": args.manifest.resolve().as_posix() if args.manifest else None,
        "split": args.split,
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
    }
    (output_root / "extraction_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if failures:
        raise RuntimeError(f"MERT extraction failed for {len(failures)} files")


if __name__ == "__main__":
    main()
