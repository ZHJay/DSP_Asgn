# -*- coding: utf-8 -*-
import os, glob, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
ROOT = str(BASE_DIR / "outputs")
ABLATIONS = ["full", "no_mel", "no_w2v2", "raw", "enhanced"]

def load_last_metrics_csv(run_dir):
    csv_path = os.path.join(run_dir, "metrics.csv")
    if not os.path.exists(csv_path):
        return None
    df = pd.read_csv(csv_path)
    last = df.iloc[-1].to_dict()
    return {
        "train_loss": float(last.get("train_loss", np.nan)),
        "eval_loss":  float(last.get("eval_loss",  np.nan)),
        "speaker_acc": float(last.get("eval_spk_acc", last.get("eval_spk_acc".upper(), np.nan))),
        "digit_acc":   float(last.get("eval_dig_acc", last.get("eval_dig_acc".upper(), np.nan))),
        "speaker_f1":  float(last.get("eval_spk_f1", np.nan)) if "eval_spk_f1" in last else np.nan,
        "digit_f1":    float(last.get("eval_dig_f1", np.nan)) if "eval_dig_f1" in last else np.nan,
    }

def try_compute_f1_from_csv(run_dir):
    # 找最后一个 eval_predictions_epoch*.csv
    files = sorted(glob.glob(os.path.join(run_dir, "eval_predictions_epoch*.csv")))
    if not files: return np.nan, np.nan
    df = pd.read_csv(files[-1])
    if not {"speaker_true","speaker_pred","digit_true","digit_pred"} <= set(df.columns):
        return np.nan, np.nan
    # 需要把标签转为类别 id（若是字符串，直接比较）
    spk_f1 = f1_score(df["speaker_true"], df["speaker_pred"], average="macro")
    dig_f1 = f1_score(df["digit_true"], df["digit_pred"], average="macro")
    return float(spk_f1), float(dig_f1)

def minmax_norm(values):
    arr = np.array(values, dtype=float)
    vmin, vmax = np.nanmin(arr), np.nanmax(arr)
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax == vmin:
        return np.ones_like(arr)
    return (arr - vmin) / (vmax - vmin)

def main():
    rows = []
    for name in ABLATIONS:
        run_dir = os.path.join(ROOT, name)
        if not os.path.isdir(run_dir):
            print(f"[WARN] missing dir: {run_dir}")
            continue
        m = load_last_metrics_csv(run_dir)
        if m is None:
            print(f"[WARN] no metrics.csv in {run_dir}")
            continue

        # F1 若缺失，尝试从 eval_predictions 计算
        if np.isnan(m["speaker_f1"]) or np.isnan(m["digit_f1"]):
            spk_f1, dig_f1 = try_compute_f1_from_csv(run_dir)
            if np.isnan(m["speaker_f1"]): m["speaker_f1"] = spk_f1 if not np.isnan(spk_f1) else m["speaker_acc"]
            if np.isnan(m["digit_f1"]):   m["digit_f1"]   = dig_f1 if not np.isnan(dig_f1)   else m["digit_acc"]

        rows.append({"model": name, **m})

    if not rows:
        print("[ERR] no data collected."); return

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(ROOT, "ablation_summary_plus.csv"), index=False, encoding="utf-8-sig")
    print(df)

    # -------- Bar: accuracy --------
    x = np.arange(len(df))
    w = 0.35
    plt.figure(figsize=(8,4))
    plt.bar(x - w/2, df["speaker_acc"], width=w, label="Speaker ACC")
    plt.bar(x + w/2, df["digit_acc"],   width=w, label="Digit ACC")
    plt.xticks(x, df["model"])
    plt.ylim(0, 1.05)
    plt.ylabel("Accuracy")
    plt.title("Ablation + Enhanced: Accuracy Comparison")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(ROOT, "ablation_accuracy_bar.png"), dpi=160)
    plt.close()

    # -------- Radar: 6 dims --------
    # dims: spk_acc, dig_acc, spk_f1, dig_f1, (1 - eval_loss)_norm, overfit_norm(小好)
    acc_spk  = df["speaker_acc"].values
    acc_dig  = df["digit_acc"].values
    f1_spk   = df["speaker_f1"].values
    f1_dig   = df["digit_f1"].values
    loss_norm= 1.0 - minmax_norm(df["eval_loss"].values)      # loss 越小越好
    gap      = df["train_loss"].values - df["eval_loss"].values
    overfit  = 1.0 - minmax_norm(gap)                         # gap 越小越好 -> 大值代表更好

    labels = ["speaker_acc","digit_acc","speaker_f1","digit_f1","1-loss(norm)","generalization"]
    data   = np.stack([acc_spk, acc_dig, f1_spk, f1_dig, loss_norm, overfit], axis=1)

    angles = np.linspace(0, 2*np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]

    plt.figure(figsize=(5.5,5.5))
    ax = plt.subplot(111, polar=True)
    for i, row in df.iterrows():
        vals = data[i].tolist()
        vals += vals[:1]
        ax.plot(angles, vals, label=row["model"])
        ax.fill(angles, vals, alpha=0.08)
    ax.set_xticks(angles[:-1]); ax.set_xticklabels(labels)
    ax.set_yticks([0.2,0.4,0.6,0.8,1.0])
    ax.set_title("Ablation + Enhanced: Radar (6 metrics)")
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.05))
    plt.tight_layout()
    plt.savefig(os.path.join(ROOT, "ablation_radar_6d.png"), dpi=160, bbox_inches="tight")
    plt.close()

    print(f"[Saved] summary csv & figures -> {ROOT}")

if __name__ == "__main__":
    main()
