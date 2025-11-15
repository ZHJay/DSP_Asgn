import importlib
import json
import os
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parent
TIME_DOMAIN_DIR = BASE_DIR.parent / "timedomain+SVM"
if str(TIME_DOMAIN_DIR) not in sys.path:
    sys.path.insert(0, str(TIME_DOMAIN_DIR))

try:
    main_module = importlib.import_module("main")
    DigitVoiceRecognizer = getattr(main_module, "DigitVoiceRecognizer")
except (ImportError, AttributeError) as exc:  # pragma: no cover
    sys.stderr.write(f"无法导入 DigitVoiceRecognizer: {exc}\n")
    sys.exit(1)


def _sorted_digit_probabilities(probabilities: Dict[str, float]) -> List[Dict[str, Any]]:
    items = [
        {"digit": int(digit), "probability": float(prob)}
        for digit, prob in probabilities.items()
    ]
    return sorted(items, key=lambda item: item["probability"], reverse=True)


def main() -> None:
    if len(sys.argv) < 2:
        sys.stderr.write("缺少音频文件路径参数\n")
        sys.exit(1)

    audio_path = Path(sys.argv[1]).resolve()
    if not audio_path.exists():
        sys.stderr.write(f"音频文件不存在: {audio_path}\n")
        sys.exit(1)

    model_path = Path(os.environ.get(
        "SVM_MODEL_PATH",
        TIME_DOMAIN_DIR / "svm_model.pkl"
    )).resolve()
    scaler_path = Path(os.environ.get(
        "SVM_SCALER_PATH",
        TIME_DOMAIN_DIR / "svm_scaler.pkl"
    )).resolve()

    if not model_path.exists():
        sys.stderr.write(f"模型文件不存在: {model_path}\n")
        sys.exit(1)

    if not scaler_path.exists():
        sys.stderr.write(f"标准化器文件不存在: {scaler_path}\n")
        sys.exit(1)

    try:
        recognizer = DigitVoiceRecognizer(
            model_path=str(model_path),
            scaler_path=str(scaler_path),
            verbose=False
        )
        prediction = recognizer.predict(str(audio_path))
    except Exception as exc:  # pragma: no cover
        sys.stderr.write(f"时域识别失败: {exc}\n")
        sys.exit(1)

    if not prediction.get("success"):
        sys.stderr.write(prediction.get("message", "识别失败"))
        sys.exit(1)

    digit_probabilities = _sorted_digit_probabilities(
        prediction.get("all_confidences", {})
    )

    time_features = prediction.get("time_feature_summary")

    payload = {
        "audio_path": str(audio_path),
        "success": True,
        "digit": prediction.get("digit"),
        "predicted_digit": prediction.get("digit"),
        "confidence": prediction.get("confidence"),
        "digit_confidence": prediction.get("confidence"),
        "digit_probabilities": digit_probabilities,
        "speaker_probabilities": [],
        "message": prediction.get("message"),
        "time_features": time_features
    }

    print(json.dumps(payload, ensure_ascii=False))


if __name__ == "__main__":
    main()
