# -*- coding: utf-8 -*-
"""
Browser-based chat UI built on top of zero-raw.py utilities with LOCAL LLM.

Pipeline per turn:
1. User records audio in the browser (Gradio microphone) or uploads audio file.
2. Server transcribes speech via Whisper.
3. Text is sent to LOCAL LLM running on 127.0.0.1:1234 (fast LLM response).
4. Reply text is synthesized by our XTTS-based TTS.
5. Browser shows chat history and plays the generated wav.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Dict, List, Tuple

import gradio as gr
import librosa
import numpy as np
import requests
import soundfile as sf
import torch
import whisper

import zero_raw  # reuse STT/TTS classes
from shared_layer import Config as SharedLayerConfig

# torch.load pickles reference __main__.Config; expose alias so checkpoints work
zero_raw.Config = SharedLayerConfig


# -----------------------------------------------------------------------------
# Globals / configs
# -----------------------------------------------------------------------------

LOCAL_LLM_URL = os.getenv("LOCAL_LLM_URL", "http://localhost:1234/v1/chat/completions")
WHISPER_MODEL_NAME = os.getenv("WHISPER_MODEL", "small")
WHISPER_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUTPUT_DIR = Path(zero_raw.tts_config.output_dir)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

whisper_model = whisper.load_model(WHISPER_MODEL_NAME, device=WHISPER_DEVICE)
tts_system = zero_raw.UnifiedTTSSystem(tts_model="xtts")


# -----------------------------------------------------------------------------
# Local LLM client
# -----------------------------------------------------------------------------

def local_llm_chat(messages: List[Dict[str, str]]) -> str:
    """
    Call local LLM running on 127.0.0.1:1234
    Compatible with OpenAI API format (e.g., LM Studio, llama.cpp server, etc.)
    """
    headers = {
        "Content-Type": "application/json",
    }
    payload = {
        "messages": messages,
        "temperature": 0.7,
        "max_tokens": -1,
        "stream": False
    }
    
    try:
        resp = requests.post(
            LOCAL_LLM_URL,
            headers=headers,
            json=payload,
            timeout=120  # 本地模型可能需要更长时间
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"].strip()
    except requests.exceptions.ConnectionError:
        raise Exception("无法连接到本地LLM服务 (localhost:1234)。请确保本地模型服务已启动。")
    except requests.exceptions.Timeout:
        raise Exception("本地LLM响应超时，请检查模型运行状态。")
    except Exception as e:
        raise Exception(f"本地LLM调用失败: {str(e)}")


# -----------------------------------------------------------------------------
# STT helper
# -----------------------------------------------------------------------------

def transcribe(audio_path: str, language: str) -> str:
    lang = "zh" if language.lower().startswith("zh") else None
    result = whisper_model.transcribe(audio_path, language=lang)
    return result.get("text", "").strip()


# -----------------------------------------------------------------------------
# Conversation states
# -----------------------------------------------------------------------------

SYSTEM_PROMPT = "You are a friendly conversational assistant."


def chat_pipeline(
    mic_audio: Tuple[int, np.ndarray],
    file_audio: str,
    reference_voice: str,
    language: str,
    history: List[Tuple[str, str]],
    messages_state: List[Dict[str, str]],
):
    if mic_audio is None and file_audio is None:
        return history, messages_state, None, "请先录音或上传音频文件再提交。"
    if not reference_voice:
        return history, messages_state, None, "请上传目标音色（参考音频）。"

    # Use file audio if provided, otherwise use microphone recording
    if file_audio is not None:
        temp_path = file_audio
    else:
        # Save microphone recording (gradio returns (sr, data))
        sr, data = mic_audio
        temp_path = OUTPUT_DIR / f"user_{int(time.time())}.wav"
        sf.write(temp_path, data, sr)

    user_text = transcribe(str(temp_path), language)
    if not user_text:
        user_text = "[未识别语音]"

    if not messages_state:
        messages_state = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages_state.append({"role": "user", "content": user_text})
    history.append((user_text, "…"))

    try:
        assistant_text = local_llm_chat(messages_state)
    except Exception as exc:
        history[-1] = (user_text, f"[本地LLM调用失败: {exc}]")
        return history, messages_state, None, f"本地LLM调用失败: {exc}"

    messages_state.append({"role": "assistant", "content": assistant_text})
    history[-1] = (user_text, assistant_text)

    out_path = OUTPUT_DIR / f"digital_reply_{int(time.time())}.wav"
    try:
        tts_system.synthesize(assistant_text, reference_voice, str(out_path), language)
    except Exception as exc:
        history[-1] = (user_text, f"[TTS 失败: {exc}]")
        return history, messages_state, None, f"TTS 失败: {exc}"

    return history, messages_state, str(out_path), "完成"


# -----------------------------------------------------------------------------
# UI
# -----------------------------------------------------------------------------

CUSTOM_CSS = """
body {
    background: #edf1ff;
    color: #0f172a;
    font-family: "Inter", "PingFang SC", sans-serif;
}
.gradio-container {
    max-width: 1200px !important;
    margin: 0 auto !important;
    padding: 30px 0 60px;
}
.hero-card {
    background: linear-gradient(135deg, #0f152b 0%, #1a2450 100%);
    border-radius: 22px;
    padding: 32px 42px;
    box-shadow: 0 25px 45px rgba(15, 23, 42, 0.35);
    margin-bottom: 26px;
    color: #ffffff;
}
.hero-card .markdown {
    color: #ffffff;
}
.the-elite {
    font-size: 13px;
    letter-spacing: 0.32em;
    color: #b3e5ff;
    text-transform: uppercase;
}
.hero-title {
    font-size: 34px;
    font-weight: 640;
    margin: 10px 0;
    color: #ffffff;
}
.hero-subtitle {
    color: #f1f5ff;
    line-height: 1.6;
}
.main-row {
    display: grid;
    grid-template-columns: 0.42fr 0.58fr;
    gap: 24px;
    align-items: start;
}
.panel {
    background: #ffffff;
    border-radius: 20px;
    padding: 24px;
    border: 1px solid #e2e8f0;
    box-shadow: 0 18px 35px rgba(15, 23, 42, 0.12);
    min-height: 520px;
    display: flex;
    flex-direction: column;
}
.panel h3 {
    margin-bottom: 16px;
    font-size: 18px;
    color: #0f172a;
}
.panel .block.svelte-1ipelgc {
    background: transparent;
}
.settings-stack > * + * {
    margin-top: 16px;
}
.chat-stack {
    display: flex;
    flex-direction: column;
    gap: 16px;
    flex: 1;
}
.chat-stack .gradio-container .component {
    background: transparent;
}
.record-row .wrap {
    flex: 1;
}
.send-btn {
    width: 100%;
    height: 54px;
    font-size: 16px;
    border-radius: 16px !important;
    background: linear-gradient(90deg, #6366f1, #8b5cf6);
    color: #fff;
}
.send-btn:hover {
    box-shadow: 0 18px 36px rgba(99, 102, 241, 0.25);
}
.status-box textarea {
    min-height: 90px !important;
}
"""


def main() -> None:
    with gr.Blocks(title="Zero-Shot Digital Human (Local LLM)", css=CUSTOM_CSS, theme=gr.themes.Soft()) as demo:
        with gr.Column(elem_classes=["hero-card"]):
            gr.Markdown("Proudly by **The Elite**", elem_classes=["the-elite"])
            gr.Markdown(
                "Zero-Shot Digital Human (本地LLM版本)",
                elem_classes=["hero-title"],
            )
            gr.Markdown(
                "对话式界面，实时完成语音采集、Whisper 转写、本地LLM回复与 XTTS 克隆音色回放，让本地数字人体验更具质感。"
                "<br>上传目标音色，按住麦克风讲话或上传音频文件，即可获得专属音色的即时语音反馈。"
                "<br><strong>⚠️ 请确保本地LLM服务运行在 localhost:1234</strong>",
                elem_classes=["hero-subtitle"],
            )

        history_state = gr.State([])
        messages_state = gr.State([])

        with gr.Row(elem_classes=["main-row"]):
            with gr.Column(elem_classes=["panel"]):
                gr.Markdown("### 设置面板")
                with gr.Column(elem_classes=["settings-stack"]):
                    reference_input = gr.Audio(label="目标音色 (wav)", type="filepath")
                    language_input = gr.Radio(
                        ["zh-cn", "en"],
                        value="zh-cn",
                        label="回复语言",
                    )
                    status_box = gr.Textbox(
                        label="系统提示",
                        interactive=False,
                        placeholder="上传参考音色后，开始录音发起对话…",
                        elem_classes=["status-box"],
                    )

            with gr.Column(elem_classes=["panel"]):
                gr.Markdown("### 对话空间")
                with gr.Column(elem_classes=["chat-stack"]):
                    history_box = gr.Chatbot(label="Chat History", height=360)
                    reply_audio = gr.Audio(label="最新克隆语音", interactive=False)
                    # 麦克风输入 - 使用正确的参数以支持 Safari
                    mic_input = gr.Audio(
                        label="按下开始录音 (点击麦克风图标)", 
                        sources=["microphone"], 
                        type="numpy",
                        streaming=False
                    )
                    # 文件上传输入
                    file_input = gr.Audio(
                        label="或上传录好的音频文件 (wav格式)",
                        sources=["upload"],
                        type="filepath"
                    )
                    submit_btn = gr.Button("发送", variant="primary", elem_classes=["send-btn"])

        submit_btn.click(
            fn=chat_pipeline,
            inputs=[mic_input, file_input, reference_input, language_input, history_state, messages_state],
            outputs=[history_box, messages_state, reply_audio, status_box],
        )

    # 使用 localhost 以支持 Safari 麦克风访问
    # Safari 要求 HTTPS 或 localhost 才能访问麦克风
    print("\n" + "="*60)
    print("🤖 本地LLM数字人系统")
    print("="*60)
    print("📋 启动检查清单:")
    print("   ✓ 本地LLM服务运行在: http://localhost:1234/v1")
    print("   ✓ Web界面将运行在: http://localhost:7860")
    print("")
    print("🎤 麦克风访问提示:")
    print("1. 请在浏览器中访问: http://localhost:7860")
    print("2. 首次使用会弹出麦克风权限请求，请点击'允许'")
    print("3. 如遇问题，请检查:")
    print("   - 系统设置 > 隐私与安全性 > 麦克风 > 确保浏览器已开启")
    print("   - 推荐使用 Chrome 浏览器以获得更好的兼容性")
    print("")
    print("💡 本地LLM服务推荐:")
    print("   - LM Studio: https://lmstudio.ai/")
    print("   - Ollama (需配置为OpenAI兼容模式)")
    print("   - llama.cpp server")
    print("="*60 + "\n")
    
    demo.launch(
        server_name="127.0.0.1", 
        server_port=7860,
        share=False,
        inbrowser=True  # 自动打开浏览器
    )


if __name__ == "__main__":
    main()
