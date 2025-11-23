#!/usr/bin/env python3
"""
智能语音锁API
提供语音验证接口，识别说话人和数字密码
"""
import argparse
import json
import os
import sys
import tempfile
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torchaudio
from transformers import AutoProcessor

# 添加父目录到路径以导入模型
FREQ_DIR = os.path.join(os.path.dirname(__file__), '..', 'frequencydomain')
sys.path.insert(0, FREQ_DIR)

try:
    from train_ablate import SpeechTransformer
except ImportError:
    print("错误: 无法导入 SpeechTransformer，请确保 train_ablate.py 在 frequencydomain 目录中")
    sys.exit(1)

# ====== 配置 ======
SAMPLE_RATE = 16000
MEL_BINS = 80
MEL_FRAMES = 128

mel_transform = torchaudio.transforms.MelSpectrogram(
    sample_rate=SAMPLE_RATE,
    n_fft=1024,
    hop_length=256,
    win_length=1024,
    n_mels=MEL_BINS,
    power=2.0,
)

def logmel(wav_1c: torch.Tensor) -> torch.Tensor:
    """wav_1c: (1, T) -> (frames, mel_bins)"""
    mel = mel_transform(wav_1c)
    mel = torch.log1p(mel).squeeze(0)
    mel = mel.transpose(0, 1)
    
    if mel.size(0) >= MEL_FRAMES:
        mel = mel[:MEL_FRAMES, :]
    else:
        pad = torch.zeros(MEL_FRAMES - mel.size(0), MEL_BINS)
        mel = torch.cat([mel, pad], dim=0)
    return mel

def softmax_t(x: torch.Tensor) -> torch.Tensor:
    e = torch.exp(x - x.max())
    return e / e.sum()

def energy_segments(wav: torch.Tensor,
                    sr: int = SAMPLE_RATE,
                    min_silence: float = 0.12,
                    min_chunk: float = 0.25,
                    th_ratio: float = 0.25) -> List[Tuple[int, int]]:
    """简单能量阈值分段"""
    x = wav.abs().numpy()
    win = int(0.02 * sr)
    if win < 1:
        win = 1
    
    kernel = np.ones(win) / win
    energy = np.convolve(x, kernel, mode="same")
    thr = energy.mean() + th_ratio * (energy.max() - energy.mean())
    
    above = energy > thr
    segments = []
    i = 0
    n = len(above)
    while i < n:
        if above[i]:
            s = i
            while i < n and above[i]:
                i += 1
            e = i
            segments.append((s, e))
        else:
            i += 1
    
    # 合并短静音
    merged = []
    min_sil = int(min_silence * sr)
    for seg in segments:
        if not merged:
            merged.append(seg)
        else:
            prev = merged[-1]
            if seg[0] - prev[1] < min_sil:
                merged[-1] = (prev[0], seg[1])
            else:
                merged.append(seg)
    
    # 过滤过短段
    min_len = int(min_chunk * sr)
    merged = [(s, e) for (s, e) in merged if (e - s) >= min_len]
    return merged

def to_chunks_for_digits(wav: torch.Tensor, digits: int) -> List[Tuple[int, int]]:
    """等长分段（兜底方案）"""
    T = wav.numel()
    step = T // digits
    chunks = []
    start = 0
    for i in range(digits - 1):
        chunks.append((start, start + step))
        start += step
    chunks.append((start, T))
    return chunks

def load_label_maps(metrics_path: str) -> Tuple[Dict[int, str], Dict[int, str]]:
    """加载标签映射"""
    with open(metrics_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    spk_map = {int(k): v for k, v in meta["label_maps"]["speaker"].items()}
    dig_map = {int(k): v for k, v in meta["label_maps"]["digit"].items()}
    return spk_map, dig_map

def verify_voice_lock(audio_path: str,
                      model_path: str,
                      metrics_path: str,
                      expected_owner: str,
                      expected_passcode: str,
                      digits: int = 4) -> Dict:
    """
    验证语音锁
    
    Args:
        audio_path: 音频文件路径
        model_path: 模型权重路径
        metrics_path: metrics.json路径
        expected_owner: 预期的主人（说话人）
        expected_passcode: 预期的密码
        digits: 预期的数字个数
        
    Returns:
        验证结果字典
    """
    # 加载标签映射
    spk_map, dig_map = load_label_maps(metrics_path)
    
    # 准备模型
    processor = AutoProcessor.from_pretrained("facebook/wav2vec2-base")
    model = SpeechTransformer(
        pretrained_model_name="facebook/wav2vec2-base",
        mel_feature_dim=MEL_BINS,
        num_speakers=len(spk_map),
        num_digits=len(dig_map),
        num_transformer_layers=4,
        num_heads=8,
        dim_feedforward=1024,
        dropout=0.1,
        freeze_encoder=False,
        ablation="full",
    )
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    
    # 读取音频
    wav, sr = torchaudio.load(audio_path)
    if wav.size(0) > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != SAMPLE_RATE:
        wav = torchaudio.transforms.Resample(sr, SAMPLE_RATE)(wav)
    wav = wav.squeeze(0)
    
    # 1. 识别说话人（整段）
    mel_full = logmel(wav.unsqueeze(0))
    proc = processor(wav.numpy(), sampling_rate=SAMPLE_RATE,
                     return_tensors="pt", padding=True, return_attention_mask=True)
    
    with torch.no_grad():
        spk_logits, _, _ = model(
            proc["input_values"],
            proc.get("attention_mask", torch.ones_like(proc["input_values"])),
            mel_full.unsqueeze(0),
            return_details=False
        )
    
    spk_prob = softmax_t(spk_logits[0])
    spk_topk = torch.topk(spk_prob, k=min(3, spk_prob.numel()))
    speaker_top3 = [
        {"name": spk_map[int(idx)], "probability": float(spk_prob[int(idx)])}
        for idx in spk_topk.indices
    ]
    
    # 2. 数字分段
    segs = energy_segments(wav, sr=SAMPLE_RATE)
    if len(segs) != digits:
        segs = to_chunks_for_digits(wav, digits)
    
    # 3. 逐段识别数字
    digits_pred = []
    digits_conf = []
    per_digit_top3 = []
    
    for i, (s, e) in enumerate(segs, start=1):
        seg_wav = wav[s:e]
        mel = logmel(seg_wav.unsqueeze(0))
        pr = processor(seg_wav.numpy(), sampling_rate=SAMPLE_RATE, return_tensors="pt",
                       padding=True, return_attention_mask=True)
        
        with torch.no_grad():
            _, dig_logits, _ = model(
                pr["input_values"],
                pr.get("attention_mask", torch.ones_like(pr["input_values"])),
                mel.unsqueeze(0),
                return_details=False
            )
        
        p = softmax_t(dig_logits[0])
        topk = torch.topk(p, k=min(3, p.numel()))
        
        idx1 = int(topk.indices[0])
        label1 = dig_map[idx1]
        prob1 = float(p[idx1])
        
        digits_pred.append(label1)
        digits_conf.append(prob1)
        
        per_digit_top3.append([
            {"digit": dig_map[int(ii)], "probability": float(p[int(ii)])}
            for ii in topk.indices
        ])
    
    recognized_digits = "".join(digits_pred)
    recognized_speaker = speaker_top3[0]["name"]
    
    # 验证逻辑
    speaker_match = (recognized_speaker == expected_owner)
    passcode_match = (recognized_digits == str(expected_passcode))
    unlock_success = speaker_match and passcode_match
    
    # 生成TTS文本
    if unlock_success:
        tts_text = f"{recognized_speaker}，欢迎回家。"
    elif not speaker_match and not passcode_match:
        tts_text = f"身份和密码均验证失败，拒绝开锁。"
    elif not speaker_match:
        tts_text = f"身份验证失败，拒绝开锁。"
    else:
        tts_text = f"密码错误，拒绝开锁。"
    
    # 返回结果
    return {
        "unlock": unlock_success,
        "speaker_match": speaker_match,
        "passcode_match": passcode_match,
        "recognized_speaker": recognized_speaker,
        "recognized_digits": recognized_digits,
        "expected_owner": expected_owner,
        "expected_passcode": str(expected_passcode),
        "speaker_confidence": speaker_top3[0]["probability"],
        "speaker_top3": speaker_top3,
        "digit_confidence": float(np.mean(digits_conf)) if digits_conf else 0.0,
        "digits_per_segment": [
            {
                "position": i,
                "recognized": digits_pred[i-1],
                "confidence": digits_conf[i-1],
                "top3": per_digit_top3[i-1]
            }
            for i in range(1, len(digits_pred) + 1)
        ],
        "tts_text": tts_text,
        "message": tts_text
    }

def main():
    parser = argparse.ArgumentParser("Smart Voice Lock API")
    parser.add_argument("--audio", required=True, help="音频文件路径")
    parser.add_argument("--model", default=os.path.join(FREQ_DIR, "outputs/best_model.pt"))
    parser.add_argument("--metrics", default=os.path.join(FREQ_DIR, "outputs/metrics.json"))
    parser.add_argument("--owner", required=True, help="预期的主人（说话人）")
    parser.add_argument("--passcode", required=True, help="预期的密码")
    parser.add_argument("--digits", type=int, default=4, help="预期数字个数")
    args = parser.parse_args()
    
    try:
        result = verify_voice_lock(
            audio_path=args.audio,
            model_path=args.model,
            metrics_path=args.metrics,
            expected_owner=args.owner,
            expected_passcode=args.passcode,
            digits=args.digits
        )
        
        # 输出JSON结果
        print(json.dumps(result, ensure_ascii=False))
        
    except Exception as e:
        error_result = {
            "unlock": False,
            "error": str(e),
            "message": f"验证失败: {str(e)}"
        }
        print(json.dumps(error_result, ensure_ascii=False))
        sys.exit(1)

if __name__ == "__main__":
    main()
