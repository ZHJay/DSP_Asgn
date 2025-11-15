import json
import os
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import torch
import torch.nn.functional as F
import torchaudio
from transformers import AutoProcessor

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parent
FREQ_DOMAIN_DIR = BASE_DIR.parent / "frequencydomain"
DEFAULT_MODEL_PATH = FREQ_DOMAIN_DIR / "outputs" / "full" / "best_model.pt"
DEFAULT_LABEL_MAP_PATH = FREQ_DOMAIN_DIR / "outputs" / "full" / "label_map.json"
DEFAULT_PROCESSOR_NAME = os.environ.get("WAV2VEC_MODEL_NAME", "facebook/wav2vec2-base")
SAMPLE_RATE = int(os.environ.get("STT_SAMPLE_RATE", "16000"))
MEL_BINS = int(os.environ.get("STT_MEL_BINS", "80"))
MEL_FRAMES = int(os.environ.get("STT_MEL_FRAMES", "128"))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPSILON = 1e-9

if str(FREQ_DOMAIN_DIR) not in sys.path:
    sys.path.insert(0, str(FREQ_DOMAIN_DIR))

SpeechTransformer = None
for candidate in ("Transformers", "speech_transformer_train"):
    try:
        module = __import__(candidate, fromlist=["SpeechTransformer"])
        SpeechTransformer = getattr(module, "SpeechTransformer")
        break
    except (ImportError, AttributeError):  # pragma: no cover - fail fast if missing
        continue

if SpeechTransformer is None:
    sys.stderr.write("无法导入 SpeechTransformer 模型定义，无法执行频域推理。\n")
    sys.exit(1)

MEL_TRANSFORM = torchaudio.transforms.MelSpectrogram(
    sample_rate=SAMPLE_RATE,
    n_fft=1024,
    hop_length=256,
    win_length=1024,
    n_mels=MEL_BINS,
    power=2.0,
)


def _load_label_sets(label_map_path: Path) -> Tuple[List[str], List[str]]:
    default_digits = [str(i) for i in range(10)]
    default_speakers = [f"speaker_{i}" for i in range(4)]

    if not label_map_path.exists():
        warnings.warn(f"标签映射文件不存在，使用默认标签: {label_map_path}")
        return default_digits, default_speakers

    try:
        with label_map_path.open("r", encoding="utf-8") as fp:
            payload = json.load(fp)
    except json.JSONDecodeError as exc:
        warnings.warn(f"解析标签映射失败，使用默认标签: {exc}")
        return default_digits, default_speakers

    digits = payload.get("digit") or default_digits
    speakers = payload.get("speaker") or default_speakers

    if not digits:
        digits = default_digits
    if not speakers:
        speakers = default_speakers

    return [str(label) for label in digits], [str(label) for label in speakers]


def _prepare_waveform(audio_path: Path) -> Tuple[torch.Tensor, float]:
    waveform, sr = torchaudio.load(str(audio_path))

    if waveform.size(0) > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    if sr != SAMPLE_RATE:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=SAMPLE_RATE)
        waveform = resampler(waveform)

    waveform = waveform.squeeze(0).contiguous()
    duration = waveform.numel() / float(SAMPLE_RATE)
    return waveform, duration


def _prepare_mel_tokens(waveform: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    mel_spec = MEL_TRANSFORM(waveform.unsqueeze(0))
    mel_spec = torch.log1p(mel_spec + EPSILON)
    original_frames = mel_spec.size(2)

    resized = F.interpolate(
        mel_spec,
        size=MEL_FRAMES,
        mode="linear",
        align_corners=False,
    ).squeeze(0)

    resized = resized.transpose(0, 1)  # (frames, mel_bins)

    effective_length = min(MEL_FRAMES, original_frames)
    if effective_length < MEL_FRAMES:
        resized[effective_length:, :] = 0.0

    padding_mask = torch.zeros(MEL_FRAMES, dtype=torch.bool)
    if effective_length < MEL_FRAMES:
        padding_mask[effective_length:] = True

    return resized.to(torch.float32), padding_mask


def _format_digit_probabilities(labels: Sequence[str], probabilities: Sequence[float]) -> List[Dict[str, Any]]:
    entries: List[Dict[str, Any]] = []
    for label, prob in zip(labels, probabilities):
        try:
            digit_value: Any = int(label)
        except ValueError:
            digit_value = label
        entries.append({"digit": digit_value, "probability": float(prob)})

    return sorted(entries, key=lambda item: item["probability"], reverse=True)


def _format_speaker_probabilities(labels: Sequence[str], probabilities: Sequence[float]) -> List[Dict[str, Any]]:
    entries: List[Dict[str, Any]] = []
    for label, prob in zip(labels, probabilities):
        entries.append({"speaker": label, "probability": float(prob)})

    return sorted(entries, key=lambda item: item["probability"], reverse=True)


def _load_model(model_path: Path, num_speakers: int, num_digits: int) -> SpeechTransformer:
    model = SpeechTransformer(
        pretrained_model_name=DEFAULT_PROCESSOR_NAME,
        mel_feature_dim=MEL_BINS,
        num_speakers=num_speakers,
        num_digits=num_digits,
    )

    checkpoint = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(checkpoint)
    model.to(DEVICE)
    model.eval()
    return model


def _run_inference(audio_path: Path, model_path: Path, label_map_path: Path) -> Dict[str, Any]:
    digit_labels, speaker_labels = _load_label_sets(label_map_path)

    if not model_path.exists():
        raise FileNotFoundError(f"模型文件不存在: {model_path}")

    processor = AutoProcessor.from_pretrained(DEFAULT_PROCESSOR_NAME)
    model = _load_model(model_path, num_speakers=len(speaker_labels), num_digits=len(digit_labels))

    waveform, duration = _prepare_waveform(audio_path)
    mel_tokens, mel_mask = _prepare_mel_tokens(waveform)

    processed = processor(
        waveform.numpy(),
        sampling_rate=SAMPLE_RATE,
        return_tensors="pt",
        padding=True,
        return_attention_mask=True,
    )

    input_values = processed["input_values"].to(DEVICE)
    attention_mask = processed["attention_mask"].to(DEVICE)
    mel_tensor = mel_tokens.unsqueeze(0).to(DEVICE)
    mel_padding_mask = mel_mask.unsqueeze(0).to(DEVICE)

    with torch.inference_mode():
        speaker_logits, digit_logits, _ = model(
            input_values,
            attention_mask,
            mel_tensor,
            mel_padding_mask=mel_padding_mask,
            return_details=False,
        )

    digit_probs = torch.softmax(digit_logits[0], dim=-1).cpu().tolist()
    speaker_probs = torch.softmax(speaker_logits[0], dim=-1).cpu().tolist()

    predicted_digit_index = int(torch.argmax(digit_logits, dim=-1).item())
    predicted_speaker_index = int(torch.argmax(speaker_logits, dim=-1).item())

    predicted_digit_label = digit_labels[predicted_digit_index]
    try:
        predicted_digit: Any = int(predicted_digit_label)
    except ValueError:
        predicted_digit = predicted_digit_label

    predicted_speaker = speaker_labels[predicted_speaker_index]

    result: Dict[str, Any] = {
        "audio_path": str(audio_path),
        "sample_rate": SAMPLE_RATE,
        "duration": duration,
        "predicted_digit": predicted_digit,
        "digit_confidence": float(digit_probs[predicted_digit_index]),
        "digit_probabilities": _format_digit_probabilities(digit_labels, digit_probs),
        "predicted_speaker": predicted_speaker,
        "speaker_confidence": float(speaker_probs[predicted_speaker_index]),
        "speaker_probabilities": _format_speaker_probabilities(speaker_labels, speaker_probs),
    }

    return result


def main() -> None:
    if len(sys.argv) < 2:
        sys.stderr.write("缺少音频文件路径参数\n")
        sys.exit(1)

    audio_path = Path(sys.argv[1]).resolve()
    if not audio_path.exists():
        sys.stderr.write(f"音频文件不存在: {audio_path}\n")
        sys.exit(1)

    model_path = Path(os.environ.get("STT_MODEL_PATH", str(DEFAULT_MODEL_PATH))).resolve()
    label_map_path = Path(os.environ.get("STT_LABEL_MAP_PATH", str(DEFAULT_LABEL_MAP_PATH))).resolve()

    try:
        payload = _run_inference(audio_path, model_path, label_map_path)
    except Exception as exc:  # pragma: no cover - surface unexpected errors
        sys.stderr.write(f"推理失败: {exc}\n")
        sys.exit(1)

    print(json.dumps(payload, ensure_ascii=False))


if __name__ == "__main__":
    main()
