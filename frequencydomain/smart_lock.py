import argparse, os, json, math
from typing import List, Dict, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torchaudio
from transformers import AutoProcessor

# 导入你训练用的模型类（确保 train_ablate.py 在同一目录）
from train_ablate import SpeechTransformer

# ====== 配置与工具 ======
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

def set_seed(seed: int = 42):
    import random
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def logmel(wav_1c: torch.Tensor) -> torch.Tensor:
    """wav_1c: (1, T) -> (frames, mel_bins)"""
    mel = mel_transform(wav_1c)            # (1, mel_bins, time)
    mel = torch.log1p(mel).squeeze(0)      # (mel_bins, time)
    mel = mel.transpose(0, 1)              # (time, mel_bins)
    # 统一到 MEL_FRAMES
    if mel.size(0) >= MEL_FRAMES:
        mel = mel[:MEL_FRAMES, :]
    else:
        pad = torch.zeros(MEL_FRAMES - mel.size(0), MEL_BINS)
        mel = torch.cat([mel, pad], dim=0)
    return mel

def load_label_maps(metrics_path: str) -> Tuple[Dict[int, str], Dict[int, str]]:
    with open(metrics_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    spk_map = {int(k): v for k, v in meta["label_maps"]["speaker"].items()}
    dig_map = {int(k): v for k, v in meta["label_maps"]["digit"].items()}
    return spk_map, dig_map

def softmax_t(x: torch.Tensor) -> torch.Tensor:
    e = torch.exp(x - x.max())
    return e / e.sum()

def energy_segments(wav: torch.Tensor,
                    sr: int = SAMPLE_RATE,
                    min_silence: float = 0.12,
                    min_chunk: float = 0.25,
                    th_ratio: float = 0.25) -> List[Tuple[int,int]]:
    """
    简单能量阈值分段：
    - 平滑：20 ms 窗，计算短时能量
    - 阈值：mean + th_ratio*(max-mean)
    - 合并短静音、过滤过短段
    返回样本点区间列表 [(s,e),...]
    """
    x = wav.abs().numpy()
    win = int(0.02*sr)
    if win < 1: win = 1
    # 平滑能量
    kernel = np.ones(win)/win
    energy = np.convolve(x, kernel, mode="same")
    thr = energy.mean() + th_ratio*(energy.max()-energy.mean())

    above = energy > thr
    segments = []
    i=0; n=len(above)
    while i<n:
        if above[i]:
            s=i
            while i<n and above[i]: i+=1
            e=i
            segments.append((s,e))
        else:
            i+=1

    # 合并相邻段中太短的静音
    merged=[]
    min_sil = int(min_silence*sr)
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
    min_len = int(min_chunk*sr)
    merged = [(s,e) for (s,e) in merged if (e-s)>=min_len]
    return merged

def to_chunks_for_digits(wav: torch.Tensor, digits: int) -> List[Tuple[int,int]]:
    """
    将整段 wav 切成 digits 个等长段（兜底方案）
    """
    T = wav.numel()
    step = T // digits
    chunks=[]
    start=0
    for i in range(digits-1):
        chunks.append((start, start+step))
        start += step
    chunks.append((start, T))
    return chunks

def plot_mel(mel: torch.Tensor, out_path: str, title: str):
    import matplotlib.pyplot as plt
    import seaborn as sns
    plt.figure(figsize=(7,3))
    sns.heatmap(mel.numpy().T, cmap="viridis")
    plt.title(title)
    plt.xlabel("Frames")
    plt.ylabel("Mel")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

# ====== 主流程 ======
def main():
    parser = argparse.ArgumentParser("Smart Voice Lock (full model)")
    parser.add_argument("--audio", required=True, help="path to lock.wav")
    parser.add_argument("--model", default="outputs/full/best_model.pt")
    parser.add_argument("--metrics", default="outputs/full/metrics.json")
    parser.add_argument("--passcode", default="3503")
    parser.add_argument("--digits", type=int, default=4, help="expected number of digits")
    parser.add_argument("--no-fig", action="store_true", help="disable saving figures")
    args = parser.parse_args()

    set_seed(42)

    # 读取 label map
    spk_map, dig_map = load_label_maps(args.metrics)

    # 准备处理器 + 模型（full：需要 wav2vec2 processor）
    processor = AutoProcessor.from_pretrained("facebook/wav2vec2-base")
    # 模型超参与训练保持一致
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
        ablation="full",   # 强制使用 full
    )
    model.load_state_dict(torch.load(args.model, map_location="cpu"))
    model.eval()

    # 读音频
    wav, sr = torchaudio.load(args.audio)
    if wav.size(0) > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != SAMPLE_RATE:
        wav = torchaudio.transforms.Resample(sr, SAMPLE_RATE)(wav)
    wav = wav.squeeze(0)  # (T,)

    # ===== 1) 识别说话人（整段）
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
    speaker_top = [(spk_map[int(idx)], float(spk_prob[int(idx)])) for idx in spk_topk.indices]

    # ===== 2) 数字分段（先能量VAD，再兜底等长切割） =====
    segs = energy_segments(wav, sr=SAMPLE_RATE)
    if len(segs) != args.digits:
        # 兜底：等分
        segs = to_chunks_for_digits(wav, args.digits)

    # ===== 3) 逐段识别数字 =====
    digits_pred = []
    digits_conf = []   # 每段 top1 概率
    per_digit_top3 = []  # 每段 top3 (label, prob)
    for i, (s,e) in enumerate(segs, start=1):
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
        # top1
        idx1 = int(topk.indices[0])
        label1 = dig_map[idx1]
        prob1 = float(p[idx1])
        digits_pred.append(label1)
        digits_conf.append(prob1)
        # 记录 top3
        per_digit_top3.append([(dig_map[int(ii)], float(p[int(ii)])) for ii in topk.indices])

    recognized = "".join(digits_pred)
    is_unlock = (recognized == str(args.passcode))
    user_name = speaker_top[0][0] if speaker_top else "Unknown"

    # ===== 4) 输出文本 & TTS 文本 =====
    print("\n================= Voice Smart Lock =================")
    print(f"Audio: {args.audio}")
    print(f"Expected Passcode: {args.passcode}")
    print(f"Recognized Digits: {recognized}")
    print(f"Unlock: {'SUCCESS ✅' if is_unlock else 'FAIL ❌'}")
    print("\nSpeaker Top-3:")
    for name, prob in speaker_top:
        print(f"  - {name}: {prob:.4f}")
    print("\nDigit segments Top-3 (per position):")
    for i, top3 in enumerate(per_digit_top3, start=1):
        trip = ", ".join([f"{lab}:{pr:.3f}" for lab, pr in top3])
        print(f"  Pos {i}: {trip}")

    if is_unlock:
        tts_text = f"{user_name}，欢迎回家。"
    else:
        tts_text = f"身份验证失败，拒绝开锁,家里不欢迎你哦。"

    print("\n[TTS TEXT]")
    print(tts_text)

    print("====================================================\n")

    # ===== 5) 保存 JSON 报告 =====
    save_dir = os.path.dirname(args.model) or "outputs/full"
    report = {
        "audio": args.audio,
        "expected_passcode": str(args.passcode),
        "recognized_digits": recognized,
        "unlock": bool(is_unlock),
        "speaker_top3": [{"name": n, "prob": p} for n,p in speaker_top],
        "digits_per_segment_top3": [
            [{"digit": lab, "prob": pr} for lab, pr in top3] for top3 in per_digit_top3
        ],
        "tts_text": tts_text,
    }
    report_path = os.path.join(save_dir, "lock_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"[Saved] Report -> {report_path}")

    # ===== 6) 可选保存可视化 =====
    if not args.no_fig:
        os.makedirs(save_dir, exist_ok=True)
        try:
            plot_mel(mel_full, os.path.join(save_dir, "lock_mel_full.png"),
                     title="Mel (full utterance)")
            # 每段小图
            for i, (s,e) in enumerate(segs, start=1):
                mel_seg = logmel(wav[s:e].unsqueeze(0))
                plot_mel(mel_seg, os.path.join(save_dir, f"lock_mel_seg{i}.png"),
                         title=f"Mel (segment {i})")
            print(f"[Saved] Figures -> {save_dir}")
        except Exception as ex:
            print(f"[Warn] Figure save failed: {ex}")

if __name__ == "__main__":
    main()
