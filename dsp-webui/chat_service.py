# -*- coding: utf-8 -*-
"""
Chat Service - 常驻 HTTP 服务，保持模型在内存中
使用 Flask 提供 REST API，避免每次请求都重新加载模型
"""

import json
import os
import sys
import time
from pathlib import Path

# Set Hugging Face mirror for China access
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import requests
import whisper
import torch
from flask import Flask, request, jsonify

# Import TTS from existing system
sys.path.append(str(Path(__file__).parent.parent / 'frequencydomain'))
try:
    import zero_raw
    from shared_layer import Config as SharedLayerConfig
    zero_raw.Config = SharedLayerConfig
except ImportError:
    print("无法导入 zero_raw 模块，请确保 frequencydomain 目录可访问")
    sys.exit(1)


LOCAL_LLM_URL = os.getenv("LOCAL_LLM_URL", "http://localhost:1234/v1/chat/completions")
WHISPER_MODEL_NAME = os.getenv("WHISPER_MODEL", "small")
WHISPER_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SERVICE_PORT = int(os.getenv("CHAT_SERVICE_PORT", "5001"))

app = Flask(__name__)

# 全局模型变量
whisper_model = None
tts_system = None
models_loaded = False


def load_models():
    """预加载所有模型到内存"""
    global whisper_model, tts_system, models_loaded
    
    if models_loaded:
        print("模型已加载，跳过重复加载")
        return
    
    print("="*60)
    print("开始加载模型...")
    print("="*60)
    
    start_time = time.time()
    
    # 加载 Whisper 模型
    print(f"\n[1/2] 加载 Whisper STT 模型 ({WHISPER_MODEL_NAME})...")
    whisper_start = time.time()
    whisper_model = whisper.load_model(WHISPER_MODEL_NAME, device=WHISPER_DEVICE)
    whisper_time = time.time() - whisper_start
    print(f"✅ Whisper 模型加载完成 ({whisper_time:.2f}秒)")
    
    # 加载 TTS 模型
    print("\n[2/2] 加载 XTTS TTS 模型...")
    tts_start = time.time()
    tts_system = zero_raw.UnifiedTTSSystem(tts_model="xtts")
    tts_time = time.time() - tts_start
    print(f"✅ XTTS 模型加载完成 ({tts_time:.2f}秒)")
    
    total_time = time.time() - start_time
    print("\n" + "="*60)
    print(f"✅ 所有模型加载完成！总耗时: {total_time:.2f}秒")
    print("="*60)
    
    models_loaded = True


@app.route('/health', methods=['GET'])
def health_check():
    """健康检查接口"""
    return jsonify({
        "status": "ok",
        "models_loaded": models_loaded,
        "whisper_device": WHISPER_DEVICE
    })


@app.route('/stt', methods=['POST'])
def stt_endpoint():
    """语音转文字接口"""
    try:
        if not models_loaded:
            load_models()
        
        # 获取上传的音频文件
        if 'audio' not in request.files:
            return jsonify({"error": "缺少音频文件", "success": False}), 400
        
        audio_file = request.files['audio']
        language = request.form.get('language', 'zh-cn')
        
        # 保存临时文件
        temp_path = f"/tmp/stt_temp_{int(time.time()*1000)}.wav"
        audio_file.save(temp_path)
        
        try:
            # 使用已加载的模型进行转写
            lang = "zh" if language.lower().startswith("zh") else None
            result = whisper_model.transcribe(temp_path, language=lang)
            text = result.get("text", "").strip()
            
            return jsonify({
                "text": text,
                "success": True
            })
        finally:
            # 清理临时文件
            if os.path.exists(temp_path):
                os.remove(temp_path)
                
    except Exception as e:
        return jsonify({"error": str(e), "success": False}), 500


@app.route('/llm', methods=['POST'])
def llm_endpoint():
    """调用本地 LLM 接口"""
    try:
        data = request.get_json()
        messages = data.get('messages', [])
        
        if not messages:
            return jsonify({"error": "缺少消息历史", "success": False}), 400
        
        headers = {"Content-Type": "application/json"}
        payload = {
            "messages": messages,
            "temperature": 0.7,
            "max_tokens": -1,
            "stream": False
        }
        
        resp = requests.post(LOCAL_LLM_URL, headers=headers, json=payload, timeout=120)
        resp.raise_for_status()
        response_text = resp.json()["choices"][0]["message"]["content"].strip()
        
        return jsonify({
            "response": response_text,
            "success": True
        })
        
    except requests.exceptions.ConnectionError:
        return jsonify({
            "error": "无法连接到本地LLM服务 (localhost:1234)。请确保本地模型服务已启动。",
            "success": False
        }), 503
    except requests.exceptions.Timeout:
        return jsonify({
            "error": "本地LLM响应超时，请检查模型运行状态。",
            "success": False
        }), 504
    except Exception as e:
        return jsonify({"error": f"本地LLM调用失败: {str(e)}", "success": False}), 500


@app.route('/tts', methods=['POST'])
def tts_endpoint():
    """文字转语音接口"""
    try:
        if not models_loaded:
            load_models()
        
        # 获取参数
        if 'reference' not in request.files:
            return jsonify({"error": "缺少参考音频文件", "success": False}), 400
        
        reference_file = request.files['reference']
        text = request.form.get('text', '').strip()
        output_path = request.form.get('output', '')
        language = request.form.get('language', 'zh-cn')
        
        if not text:
            return jsonify({"error": "缺少要合成的文本", "success": False}), 400
        
        if not output_path:
            return jsonify({"error": "缺少输出路径", "success": False}), 400
        
        # 保存临时参考音频
        temp_ref_path = f"/tmp/tts_ref_{int(time.time()*1000)}.wav"
        reference_file.save(temp_ref_path)
        
        try:
            # 使用已加载的模型进行合成
            tts_system.synthesize(text, temp_ref_path, output_path, language)
            
            return jsonify({
                "output": output_path,
                "success": True
            })
        finally:
            # 清理临时文件
            if os.path.exists(temp_ref_path):
                os.remove(temp_ref_path)
                
    except Exception as e:
        return jsonify({"error": str(e), "success": False}), 500


if __name__ == '__main__':
    print("\n" + "="*60)
    print("启动 Chat Service (常驻服务模式)")
    print(f"端口: {SERVICE_PORT}")
    print("="*60 + "\n")
    
    # 启动时预加载模型
    load_models()
    
    print(f"\n🚀 Chat Service 已启动，监听端口 {SERVICE_PORT}")
    print("="*60 + "\n")
    
    # 启动 Flask 服务
    app.run(host='0.0.0.0', port=SERVICE_PORT, debug=False, threaded=True)
