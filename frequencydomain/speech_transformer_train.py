import argparse
import json
import math
import os
import random
import re
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
import torchaudio
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tqdm.auto import tqdm
from transformers import AutoModel, AutoProcessor
import matplotlib.pyplot as plt
import seaborn as sns


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_filename(filename: str) -> Optional[Dict[str, any]]:
    """
    解析文件名格式: speaker-digit-index.wav
    例如: bck-3-5.wav -> {'speaker': 'bck', 'digit': 3, 'index': 5}
    """
    match = re.match(r"([a-z]+)-(\d+)-(\d+)\.wav$", filename)
    if match:
        return {
            'speaker': match.group(1),
            'digit': int(match.group(2)),
            'index': int(match.group(3))
        }
    return None


def load_audio_files_from_directory(sample_dir: str) -> List[Dict]:
    """
    从样本文件夹加载所有音频文件
    返回格式: [{'audio_path': path, 'speaker': speaker, 'digit': digit}, ...]
    """
    records = []
    
    if not os.path.exists(sample_dir):
        raise FileNotFoundError(f"样本目录不存在: {sample_dir}")
    
    # 遍历所有数字子文件夹 (0-9)
    for digit_folder in os.listdir(sample_dir):
        digit_path = os.path.join(sample_dir, digit_folder)
        if not os.path.isdir(digit_path):
            continue
        
        # 遍历该数字文件夹中的所有wav文件
        for wav_file in os.listdir(digit_path):
            if not wav_file.endswith('.wav'):
                continue
            
            parsed = parse_filename(wav_file)
            if parsed:
                full_path = os.path.join(digit_path, wav_file)
                records.append({
                    'audio_path': full_path,
                    'speaker': parsed['speaker'],
                    'digit': parsed['digit']
                })
    
    print(f"[数据加载] 从 {sample_dir} 加载了 {len(records)} 个音频文件")
    return records


class SpeechDataset(Dataset):
    """Dataset returning raw audio and mel-spectrogram tokens."""

    def __init__(
        self,
        records: Sequence[Dict],
        audio_root: str,
        sample_rate: int,
        mel_bins: int,
        mel_frames: int,
    ) -> None:
        self.records = list(records)
        self.audio_root = audio_root
        self.sample_rate = sample_rate
        self.mel_bins = mel_bins
        self.mel_frames = mel_frames
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=1024,
            hop_length=256,
            win_length=1024,
            n_mels=mel_bins,
            power=2.0,
        )

    def __len__(self) -> int:
        return len(self.records)

    def _resolve_audio_path(self, rel_path: str) -> str:
        if self.audio_root and not os.path.isabs(rel_path):
            return os.path.join(self.audio_root, rel_path)
        return rel_path

    def __getitem__(self, idx: int) -> Dict:
        item = self.records[idx]
        audio_path = self._resolve_audio_path(item["audio_path"])
        waveform, sr = torchaudio.load(audio_path)

        if waveform.size(0) > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.sample_rate)
            waveform = resampler(waveform)

        waveform = waveform.squeeze(0).contiguous()

        mel_spec = self.mel_transform(waveform.unsqueeze(0))  # (mel_bins, time)
        mel_spec = torch.log1p(mel_spec)
        original_frames = mel_spec.size(2)

        resized = F.interpolate(
            mel_spec,
            size=self.mel_frames,
            mode="linear",
            align_corners=False,
        ).squeeze(0)

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
    """Custom collator to align raw audio and mel feature tokens."""

    def __init__(self, processor: AutoProcessor, sample_rate: int) -> None:
        self.processor = processor
        self.sample_rate = sample_rate

    def __call__(self, batch: List[Dict]) -> Dict:
        audio_arrays = [sample["input_values"] for sample in batch]
        processed = self.processor(
            audio_arrays,
            sampling_rate=self.sample_rate,
            return_tensors="pt",
            padding=True,
            return_attention_mask=True,
        )

        mel_tensors = [sample["mel"] for sample in batch]
        mel_lengths = torch.tensor([sample["mel_length"] for sample in batch], dtype=torch.long)
        mel_stack = pad_sequence(mel_tensors, batch_first=True)

        mel_pad_mask = (
            torch.arange(mel_stack.size(1))[None, :].expand(len(batch), -1) >= mel_lengths[:, None]
        )

        speaker_labels = torch.tensor([sample["speaker_label"] for sample in batch], dtype=torch.long)
        digit_labels = torch.tensor([sample["digit_label"] for sample in batch], dtype=torch.long)
        audio_paths = [sample["audio_path"] for sample in batch]

        return {
            "input_values": processed["input_values"],
            "attention_mask": processed["attention_mask"],
            "mel_tokens": mel_stack,
            "mel_padding_mask": mel_pad_mask,
            "speaker": speaker_labels,
            "digit": digit_labels,
            "audio_paths": audio_paths,
        }


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 10000) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class TransformerEncoderLayerWithAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: str = "gelu",
    ) -> None:
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True,
        )

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout_ff = nn.Dropout(dropout)

        if activation == "gelu":
            self.activation = nn.GELU()
        elif activation == "relu":
            self.activation = nn.ReLU()
        else:
            raise ValueError(f"Unsupported activation: {activation}")

    def forward(
        self,
        src: torch.Tensor,
        src_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> (torch.Tensor, torch.Tensor):
        attn_output, attn_weights = self.self_attn(
            src,
            src,
            src,
            key_padding_mask=src_key_padding_mask,
            need_weights=True,
            average_attn_weights=False,
        )
        src = src + self.dropout1(attn_output)
        src = self.norm1(src)

        ff_output = self.linear2(self.dropout_ff(self.activation(self.linear1(src))))
        src = src + self.dropout2(ff_output)
        src = self.norm2(src)
        return src, attn_weights


class TransformerEncoderWithAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        num_layers: int,
        dim_feedforward: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [
                TransformerEncoderLayerWithAttention(
                    d_model=d_model,
                    nhead=nhead,
                    dim_feedforward=dim_feedforward,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        src: torch.Tensor,
        src_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> (torch.Tensor, List[torch.Tensor]):
        attn_weights: List[torch.Tensor] = []
        output = src
        for layer in self.layers:
            output, attn = layer(output, src_key_padding_mask=src_key_padding_mask)
            attn_weights.append(attn)
        output = self.norm(output)
        return output, attn_weights


class SpeechTransformer(nn.Module):
    def __init__(
        self,
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

        if freeze_encoder:
            self.wav_encoder.requires_grad_(False)

        self.mel_proj = nn.Linear(mel_feature_dim, self.hidden_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.hidden_size))
        self.positional_encoding = PositionalEncoding(self.hidden_size, dropout=dropout)
        self.transformer = TransformerEncoderWithAttention(
            d_model=self.hidden_size,
            nhead=num_heads,
            num_layers=num_transformer_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )
        self.dropout = nn.Dropout(dropout)
        self.speaker_head = nn.Linear(self.hidden_size, num_speakers)
        self.digit_head = nn.Linear(self.hidden_size, num_digits)

    def forward(
        self,
        input_values: torch.Tensor,
        attention_mask: torch.Tensor,
        mel_tokens: torch.Tensor,
        mel_padding_mask: Optional[torch.Tensor] = None,
        return_details: bool = False,
    ) -> (torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]):

        # wav2vec2 encoder
        encoder_outputs = self.wav_encoder(
            input_values=input_values,
            attention_mask=attention_mask,
            output_hidden_states=False,
        )
        audio_embeddings = encoder_outputs.last_hidden_state   # (B, wav_seq, H)

        # mel -> same dimension
        mel_projected = self.mel_proj(mel_tokens)              # (B, mel_seq, H)

        # concat modal
        combined_embeddings = torch.cat([audio_embeddings, mel_projected], dim=1)

        # CLS
        batch_size = combined_embeddings.size(0)
        cls_token = self.cls_token.expand(batch_size, -1, -1)
        encoder_input = torch.cat([cls_token, combined_embeddings], dim=1)

        # pos encoding
        encoder_input = self.positional_encoding(encoder_input)

        # ===== correct mask fix =====
        audio_len = audio_embeddings.size(1)
        mel_len   = mel_tokens.size(1)

        if mel_padding_mask is None:
            mel_padding_mask = torch.zeros(
                batch_size, mel_len, dtype=torch.bool, device=mel_tokens.device
            )

        # wav tokens 全 valid (CLS + wav part)
        wav_mask = torch.zeros(batch_size, 1 + audio_len, dtype=torch.bool, device=mel_tokens.device)

        src_key_padding_mask = torch.cat([wav_mask, mel_padding_mask], dim=1)
        # =============================

        # transformer encoder
        transformer_output, attn_weights = self.transformer(
            encoder_input, src_key_padding_mask=src_key_padding_mask
        )

        # CLS head
        cls_output = self.dropout(transformer_output[:, 0, :])
        speaker_logits = self.speaker_head(cls_output)
        digit_logits   = self.digit_head(cls_output)

        details: Dict[str, torch.Tensor] = {}
        if return_details:
            details = {
                "encoder_input": encoder_input.detach(),
                "combined_embeddings": combined_embeddings.detach(),
                "transformer_output": transformer_output.detach(),
                "attention_weights": [attn.detach() for attn in attn_weights],
            }

        return speaker_logits, digit_logits, details



def train_one_epoch(
    model: SpeechTransformer,
    data_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> Dict[str, float]:
    model.train()
    ce_loss = nn.CrossEntropyLoss()
    total_loss = 0.0
    total_samples = 0
    speaker_correct = 0
    digit_correct = 0

    progress = tqdm(data_loader, desc="Train", leave=False)
    for batch in progress:
        input_values = batch["input_values"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        mel_tokens = batch["mel_tokens"].to(device)
        mel_padding_mask = batch["mel_padding_mask"].to(device)
        speaker_labels = batch["speaker"].to(device)
        digit_labels = batch["digit"].to(device)

        optimizer.zero_grad(set_to_none=True)
        speaker_logits, digit_logits, _ = model(
            input_values,
            attention_mask,
            mel_tokens,
            mel_padding_mask=mel_padding_mask,
            return_details=False,
        )

        loss_speaker = ce_loss(speaker_logits, speaker_labels)
        loss_digit = ce_loss(digit_logits, digit_labels)
        loss = loss_speaker + loss_digit
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        batch_size = input_values.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size
        speaker_correct += (speaker_logits.argmax(dim=1) == speaker_labels).sum().item()
        digit_correct += (digit_logits.argmax(dim=1) == digit_labels).sum().item()

        progress.set_postfix({
            "loss": total_loss / max(total_samples, 1),
            "spk_acc": speaker_correct / max(total_samples, 1),
            "dig_acc": digit_correct / max(total_samples, 1),
        })

    return {
        "loss": total_loss / max(total_samples, 1),
        "speaker_accuracy": speaker_correct / max(total_samples, 1),
        "digit_accuracy": digit_correct / max(total_samples, 1),
    }


def evaluate(
    model: SpeechTransformer,
    data_loader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    model.eval()
    ce_loss = nn.CrossEntropyLoss()
    total_loss = 0.0
    total_samples = 0
    speaker_correct = 0
    digit_correct = 0

    with torch.no_grad():
        progress = tqdm(data_loader, desc="Eval", leave=False)
        for batch in progress:
            input_values = batch["input_values"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            mel_tokens = batch["mel_tokens"].to(device)
            mel_padding_mask = batch["mel_padding_mask"].to(device)
            speaker_labels = batch["speaker"].to(device)
            digit_labels = batch["digit"].to(device)

            speaker_logits, digit_logits, _ = model(
                input_values,
                attention_mask,
                mel_tokens,
                mel_padding_mask=mel_padding_mask,
                return_details=False,
            )

            loss_speaker = ce_loss(speaker_logits, speaker_labels)
            loss_digit = ce_loss(digit_logits, digit_labels)
            loss = loss_speaker + loss_digit

            batch_size = input_values.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size
            speaker_correct += (speaker_logits.argmax(dim=1) == speaker_labels).sum().item()
            digit_correct += (digit_logits.argmax(dim=1) == digit_labels).sum().item()

        if total_samples:
            progress.set_postfix({
                "loss": total_loss / total_samples,
                "spk_acc": speaker_correct / total_samples,
                "dig_acc": digit_correct / total_samples,
            })

    return {
        "loss": total_loss / max(total_samples, 1),
        "speaker_accuracy": speaker_correct / max(total_samples, 1),
        "digit_accuracy": digit_correct / max(total_samples, 1),
    }


def collect_embedding_and_attention(
    model: SpeechTransformer,
    data_loader: DataLoader,
    device: torch.device,
    output_dir: str,
    speaker_encoder: LabelEncoder,
    digit_encoder: LabelEncoder,
    max_batches: int = 1,
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    model.eval()

    collected_batches: List[Dict] = []
    attention_batches: List[List[torch.Tensor]] = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(data_loader):
            if batch_idx >= max_batches:
                break

            input_values = batch["input_values"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            mel_tokens = batch["mel_tokens"].to(device)
            mel_padding_mask = batch["mel_padding_mask"].to(device)

            speaker_logits, digit_logits, details = model(
                input_values,
                attention_mask,
                mel_tokens,
                mel_padding_mask=mel_padding_mask,
                return_details=True,
            )

            collected_batches.append(
                {
                    "audio_paths": batch["audio_paths"],
                    "encoder_input": details["encoder_input"].cpu(),
                    "combined_embeddings": details["combined_embeddings"].cpu(),
                    "transformer_output": details["transformer_output"].cpu(),
                    "speaker_logits": speaker_logits.cpu(),
                    "digit_logits": digit_logits.cpu(),
                    "speaker_labels": batch["speaker"],
                    "digit_labels": batch["digit"],
                }
            )
            attention_batches.append([attn.cpu() for attn in details["attention_weights"]])

    if not collected_batches:
        print("[Analysis] No batches collected for embedding/attention export.")
        return

    embedding_path = os.path.join(output_dir, "embedding_snapshot.pt")
    torch.save({"batches": collected_batches}, embedding_path)
    attention_path = os.path.join(output_dir, "attention_weights.pt")
    torch.save(attention_batches, attention_path)

    print(f"[Analysis] Saved embedding snapshot to {embedding_path}")
    print(f"[Analysis] Saved attention weights to {attention_path}")

    first_batch = collected_batches[0]
    first_combined = first_batch["combined_embeddings"][0]  # (seq, hidden)
    first_encoder_input = first_batch["encoder_input"][0]

    print("[Embedding] Combined embedding shape:", tuple(first_batch["combined_embeddings"].shape))
    print("[Embedding] Sample token vector (combined, token 0, first 8 dims):",
          first_combined[0, :8].tolist())
    print("[Embedding] Encoder input sample (after positional encoding, CLS token, first 8 dims):",
          first_encoder_input[0, :8].tolist())

    representative_attention = attention_batches[0]
    for layer_idx, layer_attn in enumerate(representative_attention):
        # layer_attn: (batch, heads, seq, seq)
        averaged = layer_attn.mean(dim=1)[0]  # first example, mean heads
        cls_to_tokens = averaged[0]
        preview = cls_to_tokens[:8].tolist()
        print(
            f"[Attention] Layer {layer_idx} CLS attention to first 8 tokens: {preview}"
        )

    # Decode predictions for inspection
    speaker_preds = first_batch["speaker_logits"].argmax(dim=1)
    digit_preds = first_batch["digit_logits"].argmax(dim=1)
    for idx, path in enumerate(first_batch["audio_paths"]):
        speaker_true = speaker_encoder.inverse_transform(first_batch["speaker_labels"][idx : idx + 1])[0]
        speaker_pred = speaker_encoder.inverse_transform(speaker_preds[idx : idx + 1])[0]
        digit_true = digit_encoder.inverse_transform(first_batch["digit_labels"][idx : idx + 1])[0]
        digit_pred = digit_encoder.inverse_transform(digit_preds[idx : idx + 1])[0]
        print(
            f"[Analysis] Sample {idx}: {path}\n"
            f"  Speaker true/pred: {speaker_true} / {speaker_pred}\n"
            f"  Digit true/pred: {digit_true} / {digit_pred}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Speech Transformer finetuning pipeline")
    parser.add_argument("--audio_root", default="", help="Root directory for relative audio paths")
    parser.add_argument("--output_dir", default="outputs", help="Directory to store logs and artifacts")
    parser.add_argument(
        "--pretrained_model",
        default="facebook/wav2vec2-base",
        help="Hugging Face pretrained speech encoder",
    )
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--mel_bins", type=int, default=80)
    parser.add_argument("--mel_frames", type=int, default=128)
    parser.add_argument("--sample_rate", type=int, default=16000)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--num_transformer_layers", type=int, default=4)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--ffn_dim", type=int, default=1024)
    parser.add_argument("--freeze_encoder", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--analysis_batches", type=int, default=1)
    parser.add_argument("--path_col", default="audio_path")
    parser.add_argument("--speaker_col", default="speaker")
    parser.add_argument("--digit_col", default="digit")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)

    records = load_audio_files_from_directory(args.audio_root)

    print(f"[Data] Scanning dataset from {args.audio_root}")
    records = load_audio_files_from_directory(args.audio_root)
    metadata = pd.DataFrame(records)

    speaker_encoder = LabelEncoder()
    digit_encoder = LabelEncoder()
    metadata["speaker_label"] = speaker_encoder.fit_transform(metadata["speaker"])
    metadata["digit_label"] = digit_encoder.fit_transform(metadata["digit"])

    train_df, test_df = train_test_split(
        metadata,
        test_size=0.1,
        random_state=args.seed,
        stratify=metadata["speaker_label"],
    )


    speaker_encoder = LabelEncoder()
    digit_encoder = LabelEncoder()
    metadata["speaker_label"] = speaker_encoder.fit_transform(metadata["speaker"])
    metadata["digit_label"] = digit_encoder.fit_transform(metadata["digit"])

    train_df, test_df = train_test_split(
        metadata,
        test_size=0.1,
        random_state=args.seed,
        stratify=metadata["speaker_label"],
    )

    train_dataset = SpeechDataset(
        train_df.to_dict("records"),
        audio_root=args.audio_root,
        sample_rate=args.sample_rate,
        mel_bins=args.mel_bins,
        mel_frames=args.mel_frames,
    )

    test_dataset = SpeechDataset(
        test_df.to_dict("records"),
        audio_root=args.audio_root,
        sample_rate=args.sample_rate,
        mel_bins=args.mel_bins,
        mel_frames=args.mel_frames,
    )

    processor = AutoProcessor.from_pretrained(args.pretrained_model)
    collator = SpeechBatchCollator(processor, sample_rate=args.sample_rate)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collator,
    )

    eval_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collator,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Setup] Using device: {device}")

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
    )
    model.to(device)

    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    history: List[Dict] = []
    best_eval_speaker = 0.0
    best_eval_digit = 0.0

    for epoch in range(1, args.epochs + 1):
        print(f"\n[Epoch {epoch}/{args.epochs}] Starting training...")
        train_metrics = train_one_epoch(model, train_loader, optimizer, device)
        eval_metrics = evaluate(model, eval_loader, device)

        print(
            f"[Epoch {epoch}] Train loss {train_metrics['loss']:.4f} | "
            f"Train speaker acc {train_metrics['speaker_accuracy']:.4f} | "
            f"Train digit acc {train_metrics['digit_accuracy']:.4f}"
        )
        print(
            f"[Epoch {epoch}] Eval loss {eval_metrics['loss']:.4f} | "
            f"Eval speaker acc {eval_metrics['speaker_accuracy']:.4f} | "
            f"Eval digit acc {eval_metrics['digit_accuracy']:.4f}"
        )

        history.append(
            {
                "epoch": epoch,
                "train": train_metrics,
                "eval": eval_metrics,
            }
        )

        if eval_metrics["speaker_accuracy"] > best_eval_speaker or eval_metrics["digit_accuracy"] > best_eval_digit:
            best_eval_speaker = max(best_eval_speaker, eval_metrics["speaker_accuracy"])
            best_eval_digit = max(best_eval_digit, eval_metrics["digit_accuracy"])
            model_path = os.path.join(args.output_dir, "best_model.pt")
            torch.save(model.state_dict(), model_path)
            print(f"[Checkpoint] Saved best model to {model_path}")

    metrics_path = os.path.join(args.output_dir, "metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as fp:
        json.dump(
            {
                "history": history,
                "label_maps": {
                    "speaker": {int(i): str(label) for i, label in enumerate(speaker_encoder.classes_)},
                    "digit": {int(i): str(label) for i, label in enumerate(digit_encoder.classes_)},
                },
            },
            fp,
            indent=2,
            ensure_ascii=False,
        )
    print(f"[Metrics] Saved metrics to {metrics_path}")

    collect_embedding_and_attention(
        model,
        eval_loader,
        device,
        output_dir=args.output_dir,
        speaker_encoder=speaker_encoder,
        digit_encoder=digit_encoder,
        max_batches=args.analysis_batches,
    )


if __name__ == "__main__":
    main()

