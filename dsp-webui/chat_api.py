# -*- coding: utf-8 -*-
"""
Chat API for digital human conversation.
Handles STT (Whisper) -> LLM (Local) -> TTS (XTTS) pipeline.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import requests
import whisper
import torch

# Import TTS from existing system
sys.path.append(str(Path(__file__).parent.parent / 'frequencydomain'))
try:
    import zero_raw
    from shared_layer import Config as SharedLayerConfig
    zero_raw.Config = SharedLayerConfig
except ImportError:
    print(json.dumps({
        "error": "无法导入 zero_raw 模块，请确保 frequencydomain 目录可访问"
    }))
    sys.exit(1)


LOCAL_LLM_URL = os.getenv("LOCAL_LLM_URL", "http://localhost:1234/v1/chat/completions")
WHISPER_MODEL_NAME = os.getenv("WHISPER_MODEL", "small")
WHISPER_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 全局变量，避免重复加载模型
whisper_model = None
tts_system = None


def get_whisper_model():
    global whisper_model
    if whisper_model is None:
        whisper_model = whisper.load_model(WHISPER_MODEL_NAME, device=WHISPER_DEVICE)
    return whisper_model


def get_tts_system():
    global tts_system
    if tts_system is None:
        tts_system = zero_raw.UnifiedTTSSystem(tts_model="xtts")
    return tts_system


def transcribe_audio(audio_path: str, language: str = "zh-cn") -> str:
    """使用 Whisper 转写音频"""
    model = get_whisper_model()
    lang = "zh" if language.lower().startswith("zh") else None
    result = model.transcribe(audio_path, language=lang)
    return result.get("text", "").strip()


def call_local_llm(messages: list) -> str:
    """调用本地 LLM"""
    headers = {"Content-Type": "application/json"}
    payload = {
        "messages": messages,
        "temperature": 0.7,
        "max_tokens": -1,
        "stream": False
    }
    
    try:
        resp = requests.post(LOCAL_LLM_URL, headers=headers, json=payload, timeout=120)
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"].strip()
    except requests.exceptions.ConnectionError:
        raise Exception("无法连接到本地LLM服务 (localhost:1234)。请确保本地模型服务已启动。")
    except requests.exceptions.Timeout:
        raise Exception("本地LLM响应超时，请检查模型运行状态。")
    except Exception as e:
        raise Exception(f"本地LLM调用失败: {str(e)}")


def synthesize_speech(text: str, reference_audio: str, output_path: str, language: str = "zh-cn"):
    """使用 XTTS 合成语音"""
    tts = get_tts_system()
    tts.synthesize(text, reference_audio, output_path, language)


def main():
    parser = argparse.ArgumentParser(description="Chat API for digital human")
    parser.add_argument("--mode", required=True, choices=["stt", "llm", "tts", "full"], help="操作模式")
    parser.add_argument("--audio", help="音频文件路径（STT模式）")
    parser.add_argument("--messages", help="消息历史（JSON格式，LLM模式）")
    parser.add_argument("--text", help="要合成的文本（TTS模式）")
    parser.add_argument("--reference", help="参考音频路径（TTS模式）")
    parser.add_argument("--output", help="输出音频路径（TTS模式）")
    parser.add_argument("--language", default="zh-cn", help="语言代码")
    
    args = parser.parse_args()
    
    try:
        if args.mode == "stt":
            # 语音转文字
            if not args.audio:
                raise ValueError("STT模式需要 --audio 参数")
            text = transcribe_audio(args.audio, args.language)
            print(json.dumps({"text": text, "success": True}))
            
        elif args.mode == "llm":
            # 调用LLM
            if not args.messages:
                raise ValueError("LLM模式需要 --messages 参数")
            messages = json.loads(args.messages)
            response = call_local_llm(messages)
            print(json.dumps({"response": response, "success": True}))
            
        elif args.mode == "tts":
            # 文字转语音
            if not all([args.text, args.reference, args.output]):
                raise ValueError("TTS模式需要 --text, --reference, --output 参数")
            synthesize_speech(args.text, args.reference, args.output, args.language)
            print(json.dumps({"output": args.output, "success": True}))
            
        elif args.mode == "full":
            # 完整流程：STT -> LLM -> TTS
            if not all([args.audio, args.messages, args.reference, args.output]):
                raise ValueError("Full模式需要 --audio, --messages, --reference, --output 参数")
            
            # 1. STT
            user_text = transcribe_audio(args.audio, args.language)
            
            # 2. LLM
            messages = json.loads(args.messages)
            messages.append({"role": "user", "content": user_text})
            assistant_text = call_local_llm(messages)
            
            # 3. TTS
            synthesize_speech(assistant_text, args.reference, args.output, args.language)
            
            print(json.dumps({
                "user_text": user_text,
                "assistant_text": assistant_text,
                "output": args.output,
                "success": True
            }))
            
    except Exception as e:
        print(json.dumps({"error": str(e), "success": False}))
        sys.exit(1)


if __name__ == "__main__":
    main()
