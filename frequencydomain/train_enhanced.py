# -*- coding: utf-8 -*-
import argparse, json, math, os, random, re, csv
from typing import Dict, List, Optional, Sequence, Tuple
import numpy as np
import pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
import torchaudio
from tqdm.auto import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, f1_score
from transformers import AutoModel, AutoProcessor
import matplotlib.pyplot as plt
import seaborn as sns

# ------------------------ Utils ------------------------
def set_seed(seed: int) -> None:
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def parse_filename(filename: str) -> Optional[Dict[str, any]]:
    m = re.match(r"([a-z]+)-(\d+)-(\d+)\.wav$", filename)
    if m:
        return {'speaker': m.group(1), 'digit': int(m.group(2)), 'index': int(m.group(3))}
    return None

def load_audio_files_from_directory(sample_dir: str) -> List[Dict]:
    records = []
    if not os.path.exists(sample_dir):
        raise FileNotFoundError(f"样本目录不存在: {sample_dir}")
    for digit_folder in os.listdir(sample_dir):
        digit_path = os.path.join(sample_dir, digit_folder)
        if not os.path.isdir(digit_path): continue
        for wav_file in os.listdir(digit_path):
            if not wav_file.endswith('.wav'): continue
            parsed = parse_filename(wav_file)
            if parsed:
                full_path = os.path.join(digit_path, wav_file)
                records.append({'audio_path': full_path,
                                'speaker': parsed['speaker'],
                                'digit': parsed['digit']})
    print(f"[数据加载] 从 {sample_dir} 加载了 {len(records)} 个音频文件")
    return records

# ------------------------ Dataset / Collator ------------------------
class SpeechDataset(Dataset):
    def __init__(self, records: Sequence[Dict], audio_root: str,
                 sample_rate: int, mel_bins: int, mel_frames: int) -> None:
        self.records = list(records)
        self.audio_root = audio_root
        self.sample_rate = sample_rate
        self.mel_bins = mel_bins
        self.mel_frames = mel_frames
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate, n_fft=1024, hop_length=256, win_length=1024,
            n_mels=mel_bins, power=2.0)

    def __len__(self) -> int: return len(self.records)

    def _resolve_audio_path(self, rel_path: str) -> str:
        if self.audio_root and not os.path.isabs(rel_path):
            return os.path.join(self.audio_root, rel_path)
        return rel_path

    def __getitem__(self, idx: int) -> Dict:
        item = self.records[idx]
        audio_path = self._resolve_audio_path(item["audio_path"])
        waveform, sr = torchaudio.load(audio_path)
        if waveform.size(0) > 1: waveform = waveform.mean(dim=0, keepdim=True)
        if sr != self.sample_rate:
            waveform = torchaudio.transforms.Resample(sr, self.sample_rate)(waveform)
        waveform = waveform.squeeze(0).contiguous()  # (T,)

        mel_spec = self.mel_transform(waveform.unsqueeze(0))  # (1, mel_bins, time)
        mel_spec = torch.log1p(mel_spec)                      # log-mel
        original_frames = mel_spec.size(2)
        resized = F.interpolate(mel_spec, size=self.mel_frames,
                                mode="linear", align_corners=False).squeeze(0)  # (mel_bins, mel_frames)
        resized = resized.transpose(0, 1)  # (mel_frames, mel_bins)
        effective_length = min(self.mel_frames, original_frames)
        if effective_length < self.mel_frames:
            resized[effective_length:, :] = 0.0

        return {
            "audio_path": audio_path,
            "input_values": waveform.numpy().astype(np.float32),
            "mel": resized.to(torch.float32),
            "mel_length": effective_length,
            "speaker_label": int(item["speaker_label"]),
            "digit_label": int(item["digit_label"]),
        }

class SpeechBatchCollator:
    def __init__(self, processor: AutoProcessor, sample_rate: int) -> None:
        self.processor = processor
        self.sample_rate = sample_rate

    def __call__(self, batch: List[Dict]) -> Dict:
        audio_arrays = [s["input_values"] for s in batch]
        processed = self.processor(audio_arrays, sampling_rate=self.sample_rate,
                                   return_tensors="pt", padding=True, return_attention_mask=True)
        input_values = processed["input_values"]
        attention_mask = processed.get("attention_mask", torch.ones_like(input_values))

        mel_tensors = [s["mel"] for s in batch]
        mel_lengths = torch.tensor([s["mel_length"] for s in batch], dtype=torch.long)
        mel_stack = pad_sequence(mel_tensors, batch_first=True)  # (B, mel_frames, mel_bins)
        mel_pad_mask = (torch.arange(mel_stack.size(1))[None, :].expand(len(batch), -1)
                        >= mel_lengths[:, None])

        speaker_labels = torch.tensor([s["speaker_label"] for s in batch], dtype=torch.long)
        digit_labels = torch.tensor([s["digit_label"] for s in batch], dtype=torch.long)
        audio_paths = [s["audio_path"] for s in batch]
        return {
            "input_values": input_values,
            "attention_mask": attention_mask,
            "mel_tokens": mel_stack,
            "mel_padding_mask": mel_pad_mask,
            "speaker": speaker_labels,
            "digit": digit_labels,
            "audio_paths": audio_paths,
        }

# ------------------------ Transformer blocks ------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 10000) -> None:
        super().__init__(); self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model); pe[:, 0::2] = torch.sin(position * div); pe[:, 1::2] = torch.cos(position * div)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.size(1)]; return self.dropout(x)

class TransformerEncoderLayerWithAttention(nn.Module):
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 2048,
                 dropout: float = 0.1, activation: str = "gelu") -> None:
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model); self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout); self.dropout2 = nn.Dropout(dropout)
        self.dropout_ff = nn.Dropout(dropout)
        self.activation = nn.GELU() if activation == "gelu" else nn.ReLU()
    def forward(self, src: torch.Tensor, src_key_padding_mask: Optional[torch.Tensor] = None
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        attn_out, attn_w = self.self_attn(src, src, src,
                                          key_padding_mask=src_key_padding_mask,
                                          need_weights=True, average_attn_weights=False)
        src = self.norm1(src + self.dropout1(attn_out))
        ff = self.linear2(self.dropout_ff(self.activation(self.linear1(src))))
        src = self.norm2(src + self.dropout2(ff))
        return src, attn_w

class TransformerEncoderWithAttention(nn.Module):
    def __init__(self, d_model: int, nhead: int, num_layers: int,
                 dim_feedforward: int, dropout: float) -> None:
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerEncoderLayerWithAttention(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
    def forward(self, src: torch.Tensor, src_key_padding_mask: Optional[torch.Tensor] = None
                ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        attn_weights = []
        out = src
        for layer in self.layers:
            out, attn = layer(out, src_key_padding_mask=src_key_padding_mask)
            attn_weights.append(attn)
        return self.norm(out), attn_weights

# ------------------------ Enhanced Model ------------------------
class SpeechTransformerEnhanced(nn.Module):
    """
    改动：模态前 LayerNorm + 拼接融合；Attention Pooling 取代 CLS；其它不变。
    """
    def __init__(self,
        pretrained_model_name: str,
        mel_feature_dim: int,
        num_speakers: int,
        num_digits: int,
        num_transformer_layers: int = 4,
        num_heads: int = 8,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        freeze_encoder: bool = False,
    ) -> None:
        super().__init__()
        self.wav_encoder = AutoModel.from_pretrained(pretrained_model_name)
        self.hidden_size = self.wav_encoder.config.hidden_size
        if freeze_encoder: self.wav_encoder.requires_grad_(False)

        self.mel_proj = nn.Linear(mel_feature_dim, self.hidden_size)

        # 关键：两路特征先各自 LayerNorm（分布/尺度对齐）
        self.norm_w2v2 = nn.LayerNorm(self.hidden_size)
        self.norm_mel  = nn.LayerNorm(self.hidden_size)

        self.positional_encoding = PositionalEncoding(self.hidden_size, dropout=dropout)
        self.transformer = TransformerEncoderWithAttention(
            d_model=self.hidden_size, nhead=num_heads,
            num_layers=num_transformer_layers, dim_feedforward=dim_feedforward, dropout=dropout
        )
        self.dropout = nn.Dropout(dropout)
        # 注意力池化查询向量（替代 CLS 聚合）
        self.pool_q = nn.Parameter(torch.randn(1,1,self.hidden_size))

        self.speaker_head = nn.Linear(self.hidden_size, num_speakers)
        self.digit_head   = nn.Linear(self.hidden_size, num_digits)

    def forward(self,
        input_values: torch.Tensor,
        attention_mask: torch.Tensor,
        mel_tokens: torch.Tensor,
        mel_padding_mask: Optional[torch.Tensor] = None,
        return_details: bool = False,
    ):
        B = input_values.size(0)

        # 两路编码
        enc = self.wav_encoder(input_values=input_values,
                               attention_mask=attention_mask,
                               output_hidden_states=False)
        wav_tokens = self.norm_w2v2(enc.last_hidden_state)    # (B, Lw, H)
        mel_proj   = self.norm_mel(self.mel_proj(mel_tokens)) # (B, Lm, H)

        combined = torch.cat([wav_tokens, mel_proj], dim=1)   # (B, Lw+Lm, H)
        # mask: wav 全有效；mel 按传入 mask
        if mel_padding_mask is None:
            mel_padding_mask = torch.zeros(B, mel_tokens.size(1), dtype=torch.bool, device=mel_tokens.device)
        wav_mask = torch.zeros(B, wav_tokens.size(1), dtype=torch.bool, device=mel_tokens.device)
        src_key_padding_mask = torch.cat([wav_mask, mel_padding_mask], dim=1)  # (B, L*)

        # 位置编码 + Transformer
        enc_in = self.positional_encoding(combined)
        out, attn = self.transformer(enc_in, src_key_padding_mask=src_key_padding_mask)  # (B, L*, H)

        # Attention Pooling（对真实帧池化）
        q = self.pool_q.expand(B, -1, -1)                         # (B,1,H)
        k = v = out                                                # (B,L,H)
        scores = (q @ k.transpose(1,2)) / math.sqrt(self.hidden_size)  # (B,1,L)
        weights = torch.softmax(scores.masked_fill(src_key_padding_mask.unsqueeze(1), -1e4), dim=-1)
        pooled = (weights @ v).squeeze(1)                          # (B,H)

        feat = self.dropout(pooled)
        spk_logits = self.speaker_head(feat)
        dig_logits = self.digit_head(feat)

        details = {}
        if return_details:
            details = {
                "encoder_input": enc_in.detach(),
                "combined_embeddings": combined.detach(),
                "transformer_output": out.detach(),
                "attention_weights": [a.detach() for a in attn],
                "pool_weights": weights.detach(),
            }
        return spk_logits, dig_logits, details

# ------------------------ Train / Eval ------------------------
def train_one_epoch(model, loader, opt, device) -> Dict[str, float]:
    model.train(); ce = nn.CrossEntropyLoss()
    total_loss=0.; n=0; spk_ok=0; dig_ok=0
    pbar = tqdm(loader, desc="Train", leave=False)
    for batch in pbar:
        x = batch["input_values"].to(device)
        am = batch["attention_mask"].to(device)
        mel= batch["mel_tokens"].to(device)
        mpm= batch["mel_padding_mask"].to(device)
        ys = batch["speaker"].to(device)
        yd = batch["digit"].to(device)

        opt.zero_grad(set_to_none=True)
        ls, ld, _ = model(x, am, mel, mpm, return_details=False)
        loss = ce(ls, ys) + ce(ld, yd)
        loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step()

        bsz = x.size(0); total_loss += loss.item()*bsz; n += bsz
        spk_ok += (ls.argmax(1)==ys).sum().item()
        dig_ok += (ld.argmax(1)==yd).sum().item()
        pbar.set_postfix(loss=total_loss/max(n,1), spk_acc=spk_ok/max(n,1), dig_acc=dig_ok/max(n,1))
    return {"loss": total_loss/max(n,1), "speaker_accuracy": spk_ok/max(n,1), "digit_accuracy": dig_ok/max(n,1)}

@torch.no_grad()
def evaluate(model, loader, device, speaker_encoder:LabelEncoder, digit_encoder:LabelEncoder,
             save_pred_csv_path:str=None, confusion_png_prefix:str=None) -> Dict[str, float]:
    model.eval(); ce = nn.CrossEntropyLoss()
    total_loss=0.; n=0; spk_ok=0; dig_ok=0
    all_true_s, all_pred_s = [], []
    all_true_d, all_pred_d = [], []
    rows = []

    pbar = tqdm(loader, desc="Eval", leave=False)
    for batch in pbar:
        x = batch["input_values"].to(device)
        am = batch["attention_mask"].to(device)
        mel= batch["mel_tokens"].to(device)
        mpm= batch["mel_padding_mask"].to(device)
        ys = batch["speaker"].to(device)
        yd = batch["digit"].to(device)
        paths = batch["audio_paths"]

        ls, ld, _ = model(x, am, mel, mpm, return_details=False)
        loss = ce(ls, ys) + ce(ld, yd)
        bsz = x.size(0); total_loss += loss.item()*bsz; n += bsz
        spk = ls.softmax(dim=1); dig = ld.softmax(dim=1)

        spk_ok += (ls.argmax(1)==ys).sum().item()
        dig_ok += (ld.argmax(1)==yd).sum().item()

        for i in range(bsz):
            all_true_s.append(ys[i].item()); all_pred_s.append(ls[i].argmax().item())
            all_true_d.append(yd[i].item()); all_pred_d.append(ld[i].argmax().item())
            rows.append({
              "audio_path": paths[i],
              "speaker_true": speaker_encoder.inverse_transform([ys[i].item()])[0],
              "speaker_pred": speaker_encoder.inverse_transform([ls[i].argmax().item()])[0],
              "digit_true":   digit_encoder.inverse_transform([yd[i].item()])[0],
              "digit_pred":   digit_encoder.inverse_transform([ld[i].argmax().item()])[0],
            })

    spk_f1 = f1_score(all_true_s, all_pred_s, average="macro")
    dig_f1 = f1_score(all_true_d, all_pred_d, average="macro")

    if save_pred_csv_path:
        pd.DataFrame(rows).to_csv(save_pred_csv_path, index=False, encoding="utf-8-sig")

    # 混淆矩阵
    spk_cm = confusion_matrix(all_true_s, all_pred_s)
    dig_cm = confusion_matrix(all_true_d, all_pred_d)
    if confusion_png_prefix:
        os.makedirs(os.path.dirname(confusion_png_prefix), exist_ok=True)
        plt.figure(figsize=(5,4)); sns.heatmap(spk_cm, annot=True, fmt="d", cmap="Blues")
        plt.title("Speaker Confusion"); plt.xlabel("Pred"); plt.ylabel("True")
        plt.tight_layout(); plt.savefig(confusion_png_prefix+"_speaker.png"); plt.close()
        plt.figure(figsize=(5,4)); sns.heatmap(dig_cm, annot=True, fmt="d", cmap="Greens")
        plt.title("Digit Confusion"); plt.xlabel("Pred"); plt.ylabel("True")
        plt.tight_layout(); plt.savefig(confusion_png_prefix+"_digit.png"); plt.close()

    return {
        "loss": total_loss/max(n,1),
        "speaker_accuracy": spk_ok/max(n,1),
        "digit_accuracy": dig_ok/max(n,1),
        "speaker_f1": float(spk_f1),
        "digit_f1": float(dig_f1),
    }

# ------------------------ Main ------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio_root", required=True)
    ap.add_argument("--output_dir", default="outputs")
    ap.add_argument("--pretrained_model", default="facebook/wav2vec2-base")
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--learning_rate", type=float, default=1e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-2)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--mel_bins", type=int, default=80)
    ap.add_argument("--mel_frames", type=int, default=128)
    ap.add_argument("--sample_rate", type=int, default=16000)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--num_transformer_layers", type=int, default=4)
    ap.add_argument("--num_heads", type=int, default=8)
    ap.add_argument("--ffn_dim", type=int, default=1024)
    ap.add_argument("--freeze_encoder", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Setup] Using device: {device}  |  Run: enhanced")

    # Data
    records = load_audio_files_from_directory(args.audio_root)
    df = pd.DataFrame(records)

    # !!! 更稳的分层：对 speaker+digit 组合分层
    df["strata"] = df["speaker"].astype(str) + "_" + df["digit"].astype(str)
    speaker_encoder = LabelEncoder(); digit_encoder = LabelEncoder()
    df["speaker_label"] = speaker_encoder.fit_transform(df["speaker"])
    df["digit_label"]   = digit_encoder.fit_transform(df["digit"])

    train_df, test_df = train_test_split(
        df, test_size=0.1, random_state=args.seed, stratify=df["strata"]
    )
    train_ds = SpeechDataset(train_df.to_dict("records"), args.audio_root,
                             args.sample_rate, args.mel_bins, args.mel_frames)
    test_ds  = SpeechDataset(test_df.to_dict("records"), args.audio_root,
                             args.sample_rate, args.mel_bins, args.mel_frames)

    processor = AutoProcessor.from_pretrained(args.pretrained_model)
    collator = SpeechBatchCollator(processor, args.sample_rate)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, collate_fn=collator)
    eval_loader  = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, collate_fn=collator)

    run_dir = os.path.join(args.output_dir, "enhanced")
    os.makedirs(run_dir, exist_ok=True)

    model = SpeechTransformerEnhanced(
        pretrained_model_name=args.pretrained_model,
        mel_feature_dim=args.mel_bins,
        num_speakers=len(speaker_encoder.classes_),
        num_digits=len(digit_encoder.classes_),
        num_transformer_layers=args.num_transformer_layers,
        num_heads=args.num_heads,
        dim_feedforward=args.ffn_dim,
        dropout=args.dropout,
        freeze_encoder=args.freeze_encoder,
    ).to(device)

    # -------- 分组学习率：预训练编码器小一点 --------
    enc_params, new_params = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad: continue
        (enc_params if n.startswith("wav_encoder.") else new_params).append(p)
    optimizer = torch.optim.AdamW(
        [{"params": enc_params, "lr": args.learning_rate * 0.25},
         {"params": new_params, "lr": args.learning_rate}],
        weight_decay=args.weight_decay
    )

    history = []; best_s=0.; best_d=0.
    csv_path = os.path.join(run_dir, "metrics.csv")
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as fcsv:
        writer = csv.writer(fcsv)
        writer.writerow(["epoch","train_loss","train_spk_acc","train_dig_acc",
                         "eval_loss","eval_spk_acc","eval_dig_acc","eval_spk_f1","eval_dig_f1"])

        for ep in range(1, args.epochs+1):
            print(f"\n[Epoch {ep}/{args.epochs}]")
            tr = train_one_epoch(model, train_loader, optimizer, device)
            ev = evaluate(
                model, eval_loader, device, speaker_encoder, digit_encoder,
                save_pred_csv_path=os.path.join(run_dir, f"eval_predictions_epoch{ep}.csv"),
                confusion_png_prefix=os.path.join(run_dir, f"confusion_epoch{ep}")
            )
            print(f"[Train] loss={tr['loss']:.4f}  spk_acc={tr['speaker_accuracy']:.4f}  dig_acc={tr['digit_accuracy']:.4f}")
            print(f"[ Eval] loss={ev['loss']:.4f}  spk_acc={ev['speaker_accuracy']:.4f}  dig_acc={ev['digit_accuracy']:.4f}  spk_f1={ev['speaker_f1']:.4f}  dig_f1={ev['digit_f1']:.4f}")

            history.append({"epoch":ep,"train":tr,"eval":ev})
            writer.writerow([ep, tr['loss'], tr['speaker_accuracy'], tr['digit_accuracy'],
                             ev['loss'], ev['speaker_accuracy'], ev['digit_accuracy'],
                             ev['speaker_f1'], ev['digit_f1']])
            fcsv.flush()

            if ev["speaker_accuracy"]>best_s or ev["digit_accuracy"]>best_d:
                best_s = max(best_s, ev["speaker_accuracy"]); best_d = max(best_d, ev["digit_accuracy"])
                torch.save(model.state_dict(), os.path.join(run_dir, "best_model.pt"))
                print(f"[Checkpoint] Saved best to {os.path.join(run_dir,'best_model.pt')}")

    with open(os.path.join(run_dir,"metrics.json"),"w",encoding="utf-8") as fp:
        json.dump({
            "args": vars(args),
            "history": history,
            "label_maps": {
                "speaker": {int(i): str(lab) for i, lab in enumerate(speaker_encoder.classes_)},
                "digit":   {int(i): str(lab) for i, lab in enumerate(digit_encoder.classes_)},
            },
        }, fp, indent=2, ensure_ascii=False)

    print(f"[Done] Logs & artifacts saved to: {run_dir}")

if __name__ == "__main__":
    main()
