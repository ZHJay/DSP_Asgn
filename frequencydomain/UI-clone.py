# -*- coding: utf-8 -*-
"""
Unified STT + Zero-Shot TTS front-end.

This script reuses the shared latent STT model (shared_layer.py) and plugs it
into several off-the-shelf zero-shot TTS backends (XTTS-v2, SpeechT5, Suno Bark).
"""

from __future__ import annotations

import argparse
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import soundfile as sf
import torch

from shared_layer import (
    Config as SharedLayerConfig,
    SharedLatentModel,
    config as shared_config,
    load_and_process_audio,
)

# Align checkpoint pickling paths: legacy checkpoints expect Config under __main__
Config = SharedLayerConfig


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class TTSConfig:
    stt_checkpoint: str = "models/best_model.pth"
    tts_models_dir: str = "models/tts"
    output_dir: str = "generated_audio_zeroshot"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    xtts_sample_rate: int = 24000


# 获取脚本所在目录作为基准路径
SCRIPT_DIR = Path(__file__).resolve().parent

tts_config = TTSConfig()


# ---------------------------------------------------------------------------
# STT wrapper
# ---------------------------------------------------------------------------


class STTSystem:
    def __init__(self, model_path: Optional[str] = None, device: Optional[str] = None) -> None:
        self.device = torch.device(device or tts_config.device)
        model_path = model_path or str(SCRIPT_DIR / tts_config.stt_checkpoint)

        print("=" * 60)
        print("Loading shared latent STT model")
        print("=" * 60)

        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        self.model = SharedLatentModel(shared_config).to(self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

        self.val_digit = checkpoint.get("val_digit_acc", float("nan"))
        self.val_speaker = checkpoint.get("val_speaker_acc", float("nan"))
        print(
            f"✅ STT ready (digit acc: {self.val_digit:.2%}, speaker acc: {self.val_speaker:.2%})"
        )

    def recognize(self, audio_path: str) -> dict:
        mel_spec = load_and_process_audio(audio_path, shared_config.MAX_FRAMES)
        mel_spec = torch.from_numpy(mel_spec).unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.model(mel_spec=mel_spec, mode="audio")
            digit_probs = torch.softmax(outputs["digit_logits"], dim=1)
            speaker_probs = torch.softmax(outputs["speaker_logits"], dim=1)

        digit_idx = digit_probs.argmax(1).item()
        speaker_idx = speaker_probs.argmax(1).item()
        speaker_name = shared_config.ID_TO_SPEAKER[speaker_idx]

        return {
            "digit": digit_idx,
            "speaker": speaker_name,
            "digit_conf": digit_probs[0, digit_idx].item(),
            "speaker_conf": speaker_probs[0, speaker_idx].item(),
        }


# ---------------------------------------------------------------------------
# Zero-shot TTS backends
# ---------------------------------------------------------------------------


class XTTSZeroShot:
    def __init__(self, device: Optional[str] = None) -> None:
        self.device = torch.device(device or tts_config.device)
        print("\n" + "=" * 60)
        print("Initialising Coqui XTTS-v2")
        print("=" * 60)
        try:
            from TTS.api import TTS as CoquiTTS  # type: ignore
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("Install the `TTS` package for XTTS-v2 support") from exc

        self.tts = CoquiTTS("tts_models/multilingual/multi-dataset/xtts_v2")
        self.tts.to(self.device)
        print(f"✅ XTTS-v2 loaded on {self.device}")

    def synthesize(self, text: str, reference_audio: str, output_path: str, language: str = "zh-cn") -> str:
        self.tts.tts_to_file(
            text=text,
            speaker_wav=reference_audio,
            language=language,
            file_path=output_path,
        )
        return output_path


# ---------------------------------------------------------------------------
# Unified orchestrator
# ---------------------------------------------------------------------------


class UnifiedTTSSystem:
    def __init__(
        self,
        tts_model: str = "xtts",
        device: Optional[str] = None,
        stt_system: Optional[STTSystem] = None,
    ) -> None:
        self.device = torch.device(device or tts_config.device)
        self.stt = stt_system or STTSystem(device=str(self.device))
        self.tts_backend = XTTSZeroShot(device=str(self.device))
        self.tts_model_type = "xtts"
        (SCRIPT_DIR / tts_config.output_dir).mkdir(parents=True, exist_ok=True)

    def recognize(self, audio_path: str) -> dict:
        result = self.stt.recognize(audio_path)
        print(
            f"识别 -> 数字: {result['digit']} (置信度 {result['digit_conf']:.2%}), "
            f"说话人: {result['speaker']} (置信度 {result['speaker_conf']:.2%})"
        )
        return result

    def synthesize(
        self,
        text: str,
        reference_audio: str,
        output_path: Optional[str] = None,
        language: str = "zh-cn",
    ) -> str:
        output_path = output_path or self._default_output_path(prefix="synth")
        path = self.tts_backend.synthesize(text, reference_audio, output_path, language)
        print(f"✅ 合成完成: {path}")
        return path

    def voice_conversion(self, source_audio: str, target_audio: str, text: Optional[str] = None) -> str:
        if text is None:
            result = self.recognize(source_audio)
            text = str(result["digit"])
        output_path = self._default_output_path(prefix="converted")
        return self.synthesize(text, target_audio, output_path)

    def _default_output_path(self, prefix: str) -> str:
        timestamp = int(time.time())
        return str(SCRIPT_DIR / tts_config.output_dir / f"{prefix}_{timestamp}.wav")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Unified STT + Zero-Shot TTS")
    subparsers = parser.add_subparsers(dest="command")

    recogn_parser = subparsers.add_parser("recognize", help="Run STT on an audio file")
    recogn_parser.add_argument("--audio", required=True)

    synth_parser = subparsers.add_parser("synthesize", help="Run zero-shot TTS")
    synth_parser.add_argument("--text", required=True)
    synth_parser.add_argument("--reference", required=True, help="Reference audio for cloning")
    synth_parser.add_argument("--language", default="zh-cn")
    synth_parser.add_argument("--output")

    convert_parser = subparsers.add_parser("convert", help="Voice conversion (recognize + clone)")
    convert_parser.add_argument("--source", required=True)
    convert_parser.add_argument("--target", required=True)

    subparsers.add_parser("interactive", help="Launch simple interactive UI")

    return parser


def launch_interactive_ui() -> None:
    import threading
    import tkinter as tk
    from tkinter import filedialog, messagebox

    stt_instance = STTSystem()
    system_cache: dict[str, UnifiedTTSSystem] = {}

    def get_system(tts_name: str) -> UnifiedTTSSystem:
        key = tts_name.lower()
        if key not in system_cache:
            system_cache[key] = UnifiedTTSSystem(
                tts_model=key,
                device=str(stt_instance.device),
                stt_system=stt_instance,
            )
        return system_cache[key]

    root = tk.Tk()
    root.title("Zero-Shot TTS")

    reference_var = tk.StringVar()
    output_var = tk.StringVar(value=str(SCRIPT_DIR / tts_config.output_dir / "interactive_out.wav"))
    language_var = tk.StringVar(value="zh-cn")
    status_var = tk.StringVar(value="请选择参考音频并输入要朗读的文本。")

    text_widget = tk.Text(root, height=6, width=60)

    def browse_reference() -> None:
        path = filedialog.askopenfilename(
            title="选择参考音频",
            filetypes=[("Wave files", "*.wav"), ("All files", "*.*")],
        )
        if path:
            reference_var.set(path)

    def browse_output() -> None:
        path = filedialog.asksaveasfilename(
            title="保存输出",
            defaultextension=".wav",
            filetypes=[("Wave files", "*.wav"), ("All files", "*.*")],
        )
        if path:
            output_var.set(path)

    def set_status(message: str) -> None:
        status_var.set(message)

    def run_synthesis() -> None:
        reference = reference_var.get().strip()
        text = text_widget.get("1.0", "end").strip()
        language = language_var.get()
        output_path = output_var.get().strip() or None

        if not reference:
            messagebox.showwarning("缺少参考音频", "请先选择参考音频。")
            return
        if not os.path.isfile(reference):
            messagebox.showerror("文件不存在", f"找不到参考音频: {reference}")
            return
        if not text:
            messagebox.showwarning("缺少文本", "请输入要朗读的文字。")
            return

        synth_button.config(state="disabled")
        set_status("正在合成，请稍候...")

        def worker() -> None:
            try:
                system = get_system("xtts")
                result_path = system.synthesize(
                    text=text,
                    reference_audio=reference,
                    output_path=output_path,
                    language=language,
                )
                root.after(
                    0,
                    lambda: (
                        set_status(f"✅ 合成完成: {result_path}"),
                        messagebox.showinfo("完成", f"合成完成！\n{result_path}"),
                    ),
                )
            except Exception as exc:  # pragma: no cover - UI-specific
                root.after(
                    0,
                    lambda: (
                        set_status("❌ 合成失败"),
                        messagebox.showerror("错误", str(exc)),
                    ),
                )
            finally:
                root.after(0, lambda: synth_button.config(state="normal"))

        threading.Thread(target=worker, daemon=True).start()

    # Layout
    tk.Label(root, text="参考音频:").grid(row=0, column=0, sticky="e", padx=5, pady=5)
    tk.Entry(root, textvariable=reference_var, width=50).grid(row=0, column=1, padx=5, pady=5)
    tk.Button(root, text="浏览...", command=browse_reference).grid(row=0, column=2, padx=5, pady=5)

    tk.Label(root, text="输出路径:").grid(row=1, column=0, sticky="e", padx=5, pady=5)
    tk.Entry(root, textvariable=output_var, width=50).grid(row=1, column=1, padx=5, pady=5)
    tk.Button(root, text="保存为...", command=browse_output).grid(row=1, column=2, padx=5, pady=5)

    tk.Label(root, text="文本内容:").grid(row=2, column=0, sticky="ne", padx=5, pady=5)
    text_widget.grid(row=2, column=1, columnspan=2, padx=5, pady=5)

    tk.Label(root, text="语言:").grid(row=3, column=0, sticky="e", padx=5, pady=5)
    tk.OptionMenu(root, language_var, "zh-cn", "en").grid(row=3, column=1, sticky="w")

    synth_button = tk.Button(root, text="开始合成", command=run_synthesis)
    synth_button.grid(row=4, column=0, columnspan=4, pady=10)

    tk.Label(root, textvariable=status_var, fg="blue").grid(row=5, column=0, columnspan=4, pady=5)

    root.mainloop()


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    if args.command is None or args.command == "interactive":
        launch_interactive_ui()
        return

    if args.command == "recognize":
        system = UnifiedTTSSystem(tts_model="xtts")
        system.recognize(args.audio)
        return

    if args.command == "synthesize":
        system = UnifiedTTSSystem()
        system.synthesize(
            text=args.text,
            reference_audio=args.reference,
            output_path=args.output,
            language=args.language,
        )
        return

    if args.command == "convert":
        system = UnifiedTTSSystem()
        system.voice_conversion(source_audio=args.source, target_audio=args.target)
        return


if __name__ == "__main__":
    main()
