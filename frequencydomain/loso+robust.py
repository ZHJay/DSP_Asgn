# eval_suite.py
import argparse, os, json, numpy as np
import torch, torchaudio
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from torch.utils.data import DataLoader

# 复用你已有的代码
from train_ablate import (
    SpeechDataset, SpeechBatchCollator, SpeechTransformer,
    load_audio_files_from_directory, set_seed, evaluate
)
from transformers import AutoProcessor

SAMPLE_RATE = 16000
MEL_BINS = 80
MEL_FRAMES = 128

# ----------------------------
# 工具：加白噪
# ----------------------------
def add_white_noise(wav: torch.Tensor, snr_db: float) -> torch.Tensor:
    x = wav
    sig_p = x.pow(2).mean()
    snr = 10 ** (snr_db / 10.0)
    noise_p = sig_p / snr
    noise = torch.randn_like(x) * noise_p.sqrt()
    return x + noise

def need_processor_for_ablation(ablation: str) -> bool:
    # 与训练一致：full/no_mel 需要 processor；no_w2v2/raw 不需要
    # enhanced 当作 full
    return ablation in ["full", "no_mel"]

def infer_ablation_for_modeldir(model_dir: str, default="full") -> str:
    # 尝试从 metrics.json 读 ablation；读不到按目录名/默认推断
    metrics_json = os.path.join(model_dir, "metrics.json")
    if os.path.exists(metrics_json):
        try:
            with open(metrics_json, "r", encoding="utf-8") as f:
                meta = json.load(f)
            if "args" in meta and "ablation" in meta["args"]:
                return meta["args"]["ablation"]
        except Exception:
            pass
    name = os.path.basename(model_dir).lower()
    if name in ["full", "no_mel", "no_w2v2", "raw"]:
        return name
    if name == "enhanced":
        return "full"
    return default

# ----------------------------
# 噪声鲁棒评测
# ----------------------------
def run_noise_eval(audio_root, model_dir, pretrained_model, snrs, device):
    # 数据 & encoders
    records = load_audio_files_from_directory(audio_root)
    df = pd.DataFrame(records)
    spk_enc = LabelEncoder(); dig_enc = LabelEncoder()
    df["speaker_label"] = spk_enc.fit_transform(df["speaker"])
    df["digit_label"]   = dig_enc.fit_transform(df["digit"])
    ds = SpeechDataset(df.to_dict("records"), audio_root, SAMPLE_RATE, MEL_BINS, MEL_FRAMES)

    # 模型与 collator
    ablation = infer_ablation_for_modeldir(model_dir)
    if os.path.basename(model_dir).lower() == "enhanced":
        ablation = "full"
    use_processor = need_processor_for_ablation(ablation)
    processor = AutoProcessor.from_pretrained(pretrained_model) if use_processor else None
    collator = SpeechBatchCollator(processor, SAMPLE_RATE, need_processor=use_processor)

    model = SpeechTransformer(
        pretrained_model_name=pretrained_model,
        mel_feature_dim=MEL_BINS,
        num_speakers=len(spk_enc.classes_),
        num_digits=len(dig_enc.classes_),
        num_transformer_layers=4, num_heads=8, dim_feedforward=1024, dropout=0.1,
        freeze_encoder=False, ablation=ablation
    ).to(device)

    ckpt = os.path.join(model_dir, "best_model.pt")
    if not os.path.exists(ckpt):
        print(f"[WARN] {model_dir}: missing best_model.pt, skip noise eval.")
        return None
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()

    results = []
    for snr in snrs:
        # 噪声 collate
        def noisy_collate(batch):
            b = collator(batch)
            x = b["input_values"]
            if not torch.is_tensor(x): x = torch.tensor(x, dtype=torch.float32)
            if x.dim() == 2:
                for i in range(x.size(0)):
                    x[i] = add_white_noise(x[i], snr)
            else:
                x = add_white_noise(x, snr)
            b["input_values"] = x
            return b

        dl = DataLoader(ds, batch_size=8, shuffle=False, collate_fn=noisy_collate)
        ev = evaluate(model, dl, device, spk_enc, dig_enc)  # 返回 loss/acc/F1（你新版 evaluate 有 F1 会带上）
        print(f"[{os.path.basename(model_dir)}] SNR={snr}dB -> spk={ev['speaker_accuracy']:.3f}  dig={ev['digit_accuracy']:.3f}")
        ev = {"snr_db": snr, **ev}
        results.append(ev)

    out_json = os.path.join(model_dir, "robust_noise.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    return results

def plot_noise_compare(all_results: dict, out_png: str):
    plt.figure(figsize=(10,4))
    # Digit
    plt.subplot(1,2,1)
    for name, rows in all_results.items():
        rows = sorted(rows, key=lambda r: r["snr_db"])
        xs = [r["snr_db"] for r in rows]
        ys = [r["digit_accuracy"] for r in rows]
        plt.plot(xs, ys, marker='o', label=name)
    plt.title("Digit Accuracy vs SNR (dB)")
    plt.xlabel("SNR (dB)"); plt.ylabel("Digit Acc")
    plt.ylim(0,1.05); plt.grid(True, alpha=0.3); plt.legend()

    # Speaker
    plt.subplot(1,2,2)
    for name, rows in all_results.items():
        rows = sorted(rows, key=lambda r: r["snr_db"])
        xs = [r["snr_db"] for r in rows]
        ys = [r["speaker_accuracy"] for r in rows]
        plt.plot(xs, ys, marker='o', label=name)
    plt.title("Speaker Accuracy vs SNR (dB)")
    plt.xlabel("SNR (dB)"); plt.ylabel("Speaker Acc")
    plt.ylim(0,1.05); plt.grid(True, alpha=0.3); plt.legend()

    os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)
    plt.tight_layout(); plt.savefig(out_png, dpi=150); plt.close()
    print(f"[Saved] {out_png}")

# ----------------------------
# LOSO 评测
# ----------------------------
@torch.no_grad()
def run_loso_eval(audio_root, model_dir, pretrained_model, device):
    # 准备完整数据
    records = load_audio_files_from_directory(audio_root)
    df = pd.DataFrame(records)
    speakers = sorted(df["speaker"].unique())

    # 统一 label encoders（基于全体）
    spk_enc = LabelEncoder(); dig_enc = LabelEncoder()
    df["speaker_label"] = spk_enc.fit_transform(df["speaker"])
    df["digit_label"]   = dig_enc.fit_transform(df["digit"])

    # 模型（固定一次）
    ablation = infer_ablation_for_modeldir(model_dir)
    if os.path.basename(model_dir).lower() == "enhanced":
        ablation = "full"
    use_processor = need_processor_for_ablation(ablation)
    processor = AutoProcessor.from_pretrained(pretrained_model) if use_processor else None
    collator = SpeechBatchCollator(processor, SAMPLE_RATE, need_processor=use_processor)

    model = SpeechTransformer(
        pretrained_model_name=pretrained_model,
        mel_feature_dim=MEL_BINS,
        num_speakers=len(spk_enc.classes_),
        num_digits=len(dig_enc.classes_),
        num_transformer_layers=4, num_heads=8, dim_feedforward=1024, dropout=0.1,
        freeze_encoder=False, ablation=ablation
    ).to(device)

    ckpt = os.path.join(model_dir, "best_model.pt")
    if not os.path.exists(ckpt):
        print(f"[WARN] {model_dir}: missing best_model.pt, skip LOSO.")
        return None
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()

    fold_results = []
    for spk in speakers:
        test_df = df[df["speaker"] == spk].copy()
        test_ds  = SpeechDataset(test_df.to_dict("records"), audio_root, SAMPLE_RATE, MEL_BINS, MEL_FRAMES)
        test_dl  = DataLoader(test_ds, batch_size=8, shuffle=False, collate_fn=collator)
        ev = evaluate(model, test_dl, device, spk_enc, dig_enc)  # 返回 loss/acc/F1
        fold_results.append({"heldout_speaker": spk, **ev})
        print(f"[{os.path.basename(model_dir)}] LOSO heldout={spk} -> spk={ev['speaker_accuracy']:.3f}  dig={ev['digit_accuracy']:.3f}")

    out_json = os.path.join(model_dir, "loso.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(fold_results, f, indent=2, ensure_ascii=False)
    return fold_results

def plot_loso_bar(all_loso: dict, out_png: str):
    """
    all_loso: {model_name: [{"heldout_speaker":..., "speaker_accuracy":..., "digit_accuracy":...}, ...]}
    画：各模型在 LOSO 上的平均 speaker/digit acc 柱状图
    """
    names = []
    spk_means = []
    dig_means = []
    for name, rows in all_loso.items():
        if not rows: continue
        spk = [r["speaker_accuracy"] for r in rows]
        dig = [r["digit_accuracy"] for r in rows]
        names.append(name)
        spk_means.append(float(np.mean(spk)))
        dig_means.append(float(np.mean(dig)))

    x = np.arange(len(names))
    w = 0.35

    plt.figure(figsize=(8,4))
    plt.bar(x - w/2, spk_means, width=w, label="Speaker Acc")
    plt.bar(x + w/2, dig_means, width=w, label="Digit Acc")
    plt.xticks(x, names); plt.ylim(0,1.05)
    plt.ylabel("Accuracy"); plt.title("LOSO Average Accuracy (Higher is better)")
    plt.legend(); plt.grid(axis="y", alpha=0.3)
    os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)
    plt.tight_layout(); plt.savefig(out_png, dpi=150); plt.close()
    print(f"[Saved] {out_png}")

# ----------------------------
# 主入口
# ----------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser("Evaluation Suite: Noise + LOSO")
    ap.add_argument("--audio_root", required=True)
    ap.add_argument("--pretrained_model", default="facebook/wav2vec2-base")
    # 模型集合（不含 no_fusion）
    ap.add_argument("--models", nargs="*", default=[
        "outputs/full",
        "outputs/no_mel",
        "outputs/no_w2v2",
        "outputs/raw",
        "outputs/enhanced",
    ])
    ap.add_argument("--snrs", nargs="*", type=float, default=[30,20,10,5,0])
    ap.add_argument("--out_noise", default="outputs/robust_noise_compare.png")
    ap.add_argument("--out_loso",  default="outputs/loso_compare.png")
    args = ap.parse_args()

    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # —— 噪声鲁棒 —— #
    noise_all = {}
    for mdir in args.models:
        res = run_noise_eval(args.audio_root, mdir, args.pretrained_model, args.snrs, device)
        if res is not None:
            noise_all[os.path.basename(mdir)] = res
    if noise_all:
        plot_noise_compare(noise_all, args.out_noise)

    # —— LOSO —— #
    loso_all = {}
    for mdir in args.models:
        res = run_loso_eval(args.audio_root, mdir, args.pretrained_model, device)
        if res is not None:
            loso_all[os.path.basename(mdir)] = res
    if loso_all:
        plot_loso_bar(loso_all, args.out_loso)
