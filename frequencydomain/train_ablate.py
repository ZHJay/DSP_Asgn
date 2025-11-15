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
from sklearn.metrics import confusion_matrix
from transformers import AutoModel, AutoProcessor
import matplotlib.pyplot as plt
import seaborn as sns

# ------------------------
# Utils
# ------------------------
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

# ------------------------
# Dataset / Collator
# ------------------------
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
    def __init__(self, processor: Optional[AutoProcessor], sample_rate: int,
                 need_processor: bool) -> None:
        self.processor = processor
        self.sample_rate = sample_rate
        self.need_processor = need_processor

    def __call__(self, batch: List[Dict]) -> Dict:
        audio_arrays = [s["input_values"] for s in batch]
        if self.need_processor:
            processed = self.processor(
                audio_arrays, sampling_rate=self.sample_rate,
                return_tensors="pt", padding=True, return_attention_mask=True,
            )
            input_values = processed["input_values"]
            attention_mask = processed.get("attention_mask", torch.ones_like(input_values))
        else:
            # raw baseline: 我们只做简单 padding（右侧补零），attention 全1（不mask）
            lengths = [len(a) for a in audio_arrays]
            maxlen = max(lengths)
            padded = [np.pad(a, (0, maxlen - len(a))) for a in audio_arrays]
            input_values = torch.tensor(np.stack(padded, 0), dtype=torch.float32)
            attention_mask = torch.ones_like(input_values)

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

# ------------------------
# Transformer blocks
# ------------------------
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

# ------------------------
# Model with Ablations
# ------------------------
class RawPatchEncoder(nn.Module):
    """raw waveform -> tokens (Conv1d patching)"""
    def __init__(self, in_ch=1, hidden=256, patch_stride=320, patch_kernel=400, proj_dim=768):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_ch, hidden, kernel_size=patch_kernel, stride=patch_stride, padding=0),
            nn.ReLU(),
            nn.Conv1d(hidden, hidden, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
        )
        self.proj = nn.Linear(hidden, proj_dim)
    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        # wav: (B, T)
        x = wav.unsqueeze(1)                 # (B,1,T)
        x = self.conv(x).transpose(1,2)      # (B, tokens, hidden)
        return self.proj(x)                  # (B, tokens, proj_dim)

class SpeechTransformer(nn.Module):
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
        ablation: str = "full",   # full | no_mel | no_w2v2 | no_fusion | raw
    ) -> None:
        super().__init__()
        self.ablation = ablation
        self.use_w2v2 = (ablation in ["full","no_mel","no_fusion"])
        self.use_mel  = (ablation in ["full","no_w2v2","no_fusion"])
        self.use_raw  = (ablation == "raw")

        self.hidden_size = 768
        if self.use_w2v2:
            self.wav_encoder = AutoModel.from_pretrained(pretrained_model_name)
            self.hidden_size = self.wav_encoder.config.hidden_size
            if freeze_encoder: self.wav_encoder.requires_grad_(False)
        if self.use_mel:
            self.mel_proj = nn.Linear(mel_feature_dim, self.hidden_size)
        if self.use_raw:
            self.raw_encoder = RawPatchEncoder(proj_dim=self.hidden_size)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.hidden_size))
        self.positional_encoding = PositionalEncoding(self.hidden_size, dropout=dropout)
        self.transformer = TransformerEncoderWithAttention(
            d_model=self.hidden_size, nhead=num_heads, num_layers=num_transformer_layers,
            dim_feedforward=dim_feedforward, dropout=dropout
        )
        self.dropout = nn.Dropout(dropout)
        self.speaker_head = nn.Linear(self.hidden_size, num_speakers)
        self.digit_head   = nn.Linear(self.hidden_size, num_digits)

    def forward(self,
        input_values: torch.Tensor,         # (B, T)
        attention_mask: torch.Tensor,       # (B, T)
        mel_tokens: torch.Tensor,           # (B, mel_frames, mel_bins)
        mel_padding_mask: Optional[torch.Tensor] = None,
        return_details: bool = False,
    ):
        B = input_values.size(0)
        streams = []

        if self.use_w2v2:
            enc = self.wav_encoder(input_values=input_values,
                                   attention_mask=attention_mask,
                                   output_hidden_states=False)
            wav_tokens = enc.last_hidden_state               # (B, L1, H)
            streams.append(wav_tokens)

        if self.use_mel:
            mel_proj = self.mel_proj(mel_tokens)             # (B, L2, H)
            streams.append(mel_proj)

        if self.use_raw:
            raw_tokens = self.raw_encoder(input_values)      # (B, Lr, H)
            streams.append(raw_tokens)

        if not streams:
            raise ValueError("No input stream selected in ablation!")

        combined = torch.cat(streams, dim=1)                 # (B, L*, H)
        cls = self.cls_token.expand(B, -1, -1)
        enc_in = torch.cat([cls, combined], dim=1)
        enc_in = self.positional_encoding(enc_in)

        # ---- padding mask（只对mel部分使用mask；w2v2/raw视为有效）----
        L_wav = streams[0].size(1) if (self.use_w2v2) else 0
        L_mel = mel_tokens.size(1) if self.use_mel else 0
        L_raw = streams[-1].size(1) if (self.use_raw and not self.use_mel and not self.use_w2v2) else (streams[-1].size(1) if self.use_raw and self.use_mel and not self.use_w2v2 else (streams[-1].size(1) if self.use_raw and not self.use_mel and self.use_w2v2 else (streams[-1].size(1) if self.use_raw and self.use_mel and self.use_w2v2 else 0)))
        # 统一做法：CLS + (所有非mel流) 全有效；mel用传入mask
        non_mel_len = 1 + (L_wav if self.use_w2v2 else 0) + (L_raw if self.use_raw else 0)
        non_mel_mask = torch.zeros(B, non_mel_len, dtype=torch.bool, device=enc_in.device)
        if self.use_mel:
            if mel_padding_mask is None:
                mel_padding_mask = torch.zeros(B, L_mel, dtype=torch.bool, device=enc_in.device)
            src_key_padding_mask = torch.cat([non_mel_mask, mel_padding_mask], dim=1)
        else:
            src_key_padding_mask = non_mel_mask

        # ---- no_fusion ablation：跳过transformer，直接池化 ----
        if self.ablation == "no_fusion":
            # 简单 mean pool（去掉CLS），也可以用 attention pooling
            pooled = combined.mean(dim=1)
            logits_spk = self.speaker_head(self.dropout(pooled))
            logits_dig = self.digit_head(self.dropout(pooled))
            details = {}
            if return_details:
                details = {"encoder_input": enc_in.detach(),
                           "combined_embeddings": combined.detach(),
                           "transformer_output": pooled.detach(),
                           "attention_weights": []}
            return logits_spk, logits_dig, details

        # ---- 正常 Fusion Transformer ----
        out, attn = self.transformer(enc_in, src_key_padding_mask=src_key_padding_mask)
        cls_out = self.dropout(out[:, 0, :])
        logits_spk = self.speaker_head(cls_out)
        logits_dig = self.digit_head(cls_out)

        details = {}
        if return_details:
            details = {"encoder_input": enc_in.detach(),
                       "combined_embeddings": combined.detach(),
                       "transformer_output": out.detach(),
                       "attention_weights": [a.detach() for a in attn]}
        return logits_spk, logits_dig, details

# ------------------------
# Train / Eval
# ------------------------
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

        spk_top3 = torch.topk(spk, k=min(3, spk.size(1)), dim=1)
        dig_top3 = torch.topk(dig, k=min(3, dig.size(1)), dim=1)

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
              "speaker_top3_idx": spk_top3.indices[i].cpu().tolist(),
              "speaker_top3_prob": [float(p) for p in spk_top3.values[i].cpu()],
              "digit_top3_idx": dig_top3.indices[i].cpu().tolist(),
              "digit_top3_prob": [float(p) for p in dig_top3.values[i].cpu()],
            })

    if save_pred_csv_path:
        # 展开 top3 索引为标签
        for r in rows:
            r["speaker_top3_label"] = [speaker_encoder.inverse_transform([k])[0] for k in r["speaker_top3_idx"]]
            r["digit_top3_label"]   = [digit_encoder.inverse_transform([k])[0] for k in r["digit_top3_idx"]]
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

    return {"loss": total_loss/max(n,1), "speaker_accuracy": spk_ok/max(n,1), "digit_accuracy": dig_ok/max(n,1)}

# ------------------------
# Embedding dump (for visualize.py)
# ------------------------
@torch.no_grad()
def collect_embedding_and_attention(model, loader, device, output_dir,
                                    speaker_encoder, digit_encoder, max_batches=1):
    os.makedirs(output_dir, exist_ok=True)
    model.eval()
    collected_batches: List[Dict] = []; attention_batches: List[List[torch.Tensor]] = []
    for bi, batch in enumerate(loader):
        if bi >= max_batches: break
        x = batch["input_values"].to(device)
        am = batch["attention_mask"].to(device)
        mel= batch["mel_tokens"].to(device)
        mpm= batch["mel_padding_mask"].to(device)
        ls, ld, details = model(x, am, mel, mpm, return_details=True)
        collected_batches.append({
            "audio_paths": batch["audio_paths"],
            "encoder_input": details["encoder_input"].cpu(),
            "combined_embeddings": details["combined_embeddings"].cpu(),
            "transformer_output": details["transformer_output"].cpu(),
            "speaker_logits": ls.cpu(),
            "digit_logits": ld.cpu(),
            "speaker_labels": batch["speaker"],
            "digit_labels": batch["digit"],
        })
        attention_batches.append([a.cpu() for a in details["attention_weights"]])

    if not collected_batches: print("[Analysis] No batches collected."); return
    torch.save({"batches": collected_batches}, os.path.join(output_dir, "embedding_snapshot.pt"))
    torch.save(attention_batches, os.path.join(output_dir, "attention_weights.pt"))
    print(f"[Analysis] Saved embedding & attention to {output_dir}")

# ------------------------
# Main
# ------------------------
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
    ap.add_argument("--analysis_batches", type=int, default=1)
    ap.add_argument("--ablation", choices=["full","no_mel","no_w2v2","no_fusion","raw"], default="full")
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Setup] Using device: {device}  |  Ablation: {args.ablation}")

    # Data scan
    records = load_audio_files_from_directory(args.audio_root)
    df = pd.DataFrame(records)

    speaker_encoder = LabelEncoder(); digit_encoder = LabelEncoder()
    df["speaker_label"] = speaker_encoder.fit_transform(df["speaker"])
    df["digit_label"]   = digit_encoder.fit_transform(df["digit"])

    train_df, test_df = train_test_split(df, test_size=0.1, random_state=args.seed,
                                         stratify=df["speaker_label"])
    train_ds = SpeechDataset(train_df.to_dict("records"), args.audio_root,
                             args.sample_rate, args.mel_bins, args.mel_frames)
    test_ds  = SpeechDataset(test_df.to_dict("records"), args.audio_root,
                             args.sample_rate, args.mel_bins, args.mel_frames)

    need_processor = (args.ablation in ["full","no_mel","no_fusion"])
    processor = AutoProcessor.from_pretrained(args.pretrained_model) if need_processor else None
    collator = SpeechBatchCollator(processor, args.sample_rate, need_processor=need_processor)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, collate_fn=collator)
    eval_loader  = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, collate_fn=collator)

    # Output dir per ablation
    run_dir = os.path.join(args.output_dir, args.ablation)
    os.makedirs(run_dir, exist_ok=True)

    # Model
    model = SpeechTransformer(
        pretrained_model_name=args.pretrained_model,
        mel_feature_dim=args.mel_bins,
        num_speakers=len(speaker_encoder.classes_),
        num_digits=len(digit_encoder.classes_),
        num_transformer_layers=args.num_transformer_layers,
        num_heads=args.num_heads,
        dim_feedforward=args.ffn_dim,
        dropout=args.dropout,
        freeze_encoder=args.freeze_encoder,
        ablation=args.ablation,
    ).to(device)

    opt = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),
                            lr=args.learning_rate, weight_decay=args.weight_decay)

    history = []; best_s=0.; best_d=0.
    csv_path = os.path.join(run_dir, "metrics.csv")
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as fcsv:
        writer = csv.writer(fcsv)
        writer.writerow(["epoch","train_loss","train_spk_acc","train_dig_acc",
                         "eval_loss","eval_spk_acc","eval_dig_acc"])

        for ep in range(1, args.epochs+1):
            print(f"\n[Epoch {ep}/{args.epochs}]")
            tr = train_one_epoch(model, train_loader, opt, device)
            ev = evaluate(
                model, eval_loader, device, speaker_encoder, digit_encoder,
                save_pred_csv_path=os.path.join(run_dir, f"eval_predictions_epoch{ep}.csv"),
                confusion_png_prefix=os.path.join(run_dir, f"confusion_epoch{ep}")
            )
            print(f"[Train] loss={tr['loss']:.4f}  spk_acc={tr['speaker_accuracy']:.4f}  dig_acc={tr['digit_accuracy']:.4f}")
            print(f"[ Eval] loss={ev['loss']:.4f}  spk_acc={ev['speaker_accuracy']:.4f}  dig_acc={ev['digit_accuracy']:.4f}")

            history.append({"epoch":ep,"train":tr,"eval":ev})
            writer.writerow([ep, tr['loss'], tr['speaker_accuracy'], tr['digit_accuracy'],
                             ev['loss'], ev['speaker_accuracy'], ev['digit_accuracy']])
            fcsv.flush()

            if ev["speaker_accuracy"]>best_s or ev["digit_accuracy"]>best_d:
                best_s = max(best_s, ev["speaker_accuracy"]); best_d = max(best_d, ev["digit_accuracy"])
                torch.save(model.state_dict(), os.path.join(run_dir, "best_model.pt"))
                print(f"[Checkpoint] Saved best to {os.path.join(run_dir,'best_model.pt')}")

    # Save metrics & label maps
    with open(os.path.join(run_dir,"metrics.json"),"w",encoding="utf-8") as fp:
        json.dump({
            "args": vars(args),
            "history": history,
            "label_maps": {
                "speaker": {int(i): str(lab) for i, lab in enumerate(speaker_encoder.classes_)},
                "digit":   {int(i): str(lab) for i, lab in enumerate(digit_encoder.classes_)},
            },
        }, fp, indent=2, ensure_ascii=False)
    with open(os.path.join(run_dir,"label_map.json"),"w",encoding="utf-8") as fp:
        json.dump({
            "speaker": list(map(str, speaker_encoder.classes_)),
            "digit":   list(map(str, digit_encoder.classes_)),
        }, fp, indent=2, ensure_ascii=False)

    # dump a small embedding/attention snapshot for visualize.py
    collect_embedding_and_attention(
        model, eval_loader, device, output_dir=run_dir,
        speaker_encoder=speaker_encoder, digit_encoder=digit_encoder,
        max_batches=1
    )
    print(f"[Done] Logs & artifacts saved to: {run_dir}")

if __name__ == "__main__":
    main()
