# -*- coding: utf-8 -*-
"""
Browser-based chat UI built on top of zero-raw.py utilities.

Pipeline per turn:
1. User records audio in the browser (Gradio microphone).
2. Server transcribes speech via Whisper.
3. Text is sent to DeepSeek Chat (fast LLM response).
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

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "sk-0f16d4357768478e9b50935fb7cfd614")
WHISPER_MODEL_NAME = os.getenv("WHISPER_MODEL", "small")
WHISPER_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUTPUT_DIR = Path(zero_raw.tts_config.output_dir)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

whisper_model = whisper.load_model(WHISPER_MODEL_NAME, device=WHISPER_DEVICE)
tts_system = zero_raw.UnifiedTTSSystem(tts_model="xtts")


# -----------------------------------------------------------------------------
# DeepSeek client
# -----------------------------------------------------------------------------

def deepseek_chat(messages: List[Dict[str, str]]) -> str:
    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {"model": "deepseek-chat", "messages": messages, "stream": False}
    resp = requests.post(
        os.getenv("DEEPSEEK_API_URL", "https://api.deepseek.com/v1/chat/completions"),
        headers=headers,
        json=payload,
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"].strip()


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
    reference_voice: str,
    language: str,
    history: List[Tuple[str, str]],
    messages_state: List[Dict[str, str]],
):
    if mic_audio is None:
        return history, messages_state, None, "请先录音再提交。"
    if not reference_voice:
        return history, messages_state, None, "请上传目标音色（参考音频）。"

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
        assistant_text = deepseek_chat(messages_state)
    except Exception as exc:
        history[-1] = (user_text, f"[DeepSeek 调用失败: {exc}]")
        return history, messages_state, None, f"DeepSeek API 调用失败: {exc}"

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
    with gr.Blocks(title="Zero-Shot Digital Human", css=CUSTOM_CSS, theme=gr.themes.Soft()) as demo:
        with gr.Column(elem_classes=["hero-card"]):
            gr.Markdown("Proudly by **The Elite**", elem_classes=["the-elite"])
            gr.Markdown(
                "Zero-Shot Digital Human",
                elem_classes=["hero-title"],
            )
            gr.Markdown(
                "对话式界面，实时完成语音采集、Whisper 转写、DeepSeek 回复与 XTTS 克隆音色回放，让本地数字人体验更具质感。"
                "<br>上传目标音色，按住麦克风讲话，即可获得专属音色的即时语音反馈。",
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
                    mic_input = gr.Audio(label="按下开始录音", sources=["microphone"], type="numpy")
                    submit_btn = gr.Button("发送", variant="primary", elem_classes=["send-btn"])

        submit_btn.click(
            fn=chat_pipeline,
            inputs=[mic_input, reference_input, language_input, history_state, messages_state],
            outputs=[history_box, messages_state, reply_audio, status_box],
        )

    demo.launch(server_name="0.0.0.0", server_port=7860)


if __name__ == "__main__":
    main()
