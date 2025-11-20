#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
TTS API for voice cloning using XTTS-v2
"""

import argparse
import sys
import warnings
from pathlib import Path

import torch

warnings.filterwarnings("ignore")

# 添加 frequencydomain 目录到路径
BASE_DIR = Path(__file__).resolve().parent
FREQ_DOMAIN_DIR = BASE_DIR.parent / "frequencydomain"
sys.path.insert(0, str(FREQ_DOMAIN_DIR))

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def synthesize_voice(reference_audio: str, text: str, output_path: str, language: str = "zh-cn") -> None:
    """使用 XTTS-v2 进行声音克隆"""
    try:
        from TTS.api import TTS as CoquiTTS
    except ImportError as exc:
        raise RuntimeError("需要安装 TTS 包才能使用 XTTS-v2") from exc

    print(f"正在加载 XTTS-v2 模型到 {DEVICE}...", file=sys.stderr)
    tts = CoquiTTS("tts_models/multilingual/multi-dataset/xtts_v2")
    tts.to(DEVICE)

    print(f"正在合成语音...", file=sys.stderr)
    tts.tts_to_file(
        text=text,
        speaker_wav=reference_audio,
        language=language,
        file_path=output_path,
    )
    
    print(f"✅ 合成完成: {output_path}", file=sys.stderr)


def main() -> None:
    parser = argparse.ArgumentParser(description="Voice cloning TTS API")
    parser.add_argument("--reference", required=True, help="参考音频文件路径")
    parser.add_argument("--text", required=True, help="要合成的文本")
    parser.add_argument("--output", required=True, help="输出音频文件路径")
    parser.add_argument("--language", default="zh-cn", help="语言代码 (zh-cn 或 en)")
    
    args = parser.parse_args()
    
    reference_path = Path(args.reference)
    if not reference_path.exists():
        print(f"错误: 参考音频文件不存在: {reference_path}", file=sys.stderr)
        sys.exit(1)
    
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        synthesize_voice(
            str(reference_path),
            args.text,
            str(output_path),
            args.language
        )
        print("SUCCESS")  # 输出成功标记给 Node.js
    except Exception as exc:
        print(f"合成失败: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
