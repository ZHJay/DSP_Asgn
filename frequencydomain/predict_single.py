import argparse
import torch
import torchaudio
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from transformers import AutoProcessor
from speech_transformer_train import SpeechTransformer
mel_transform = torchaudio.transforms.MelSpectrogram(
    sample_rate=16000,
    n_fft=1024,
    hop_length=256,
    win_length=1024,
    n_mels=80,
    power=2.0,
)

def extract_mel(path):
    wav, sr = torchaudio.load(path)
    if wav.size(0) > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != 16000:
        wav = torchaudio.transforms.Resample(sr,16000)(wav)
    mel = mel_transform(wav)
    mel = torch.log1p(mel)
    mel = mel.squeeze(0).transpose(0,1)       # (frames,nmels)
    return wav.squeeze(0), mel

def plot_and_save_mel(mel, outpath):
    plt.figure(figsize=(6,4))
    sns.heatmap(mel.numpy().T, cmap="viridis")   # T因为视觉上mel_rows作为y轴习惯
    plt.title("Mel Spectrogram")
    plt.savefig(outpath)
    plt.close()

def plot_embedding(embedding, path):
    plt.figure(figsize=(6,8))
    sns.heatmap(embedding.numpy(), cmap="viridis")
    plt.title("Embedding Heatmap")
    plt.savefig(path)
    plt.close()

def plot_attention(attn, path):
    # attn list: layers * (batch,heads,seq,seq)
    last = attn[-1]  # last layer
    avg = last.mean(dim=1)[0] # first sample mean head
    plt.figure(figsize=(6,6))
    sns.heatmap(avg.numpy(), cmap="viridis")
    plt.title("Last Layer CLS Attention")
    plt.savefig(path)
    plt.close()

def softmax(x):
    e = torch.exp(x - torch.max(x))
    return e / e.sum()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", required=True)
    parser.add_argument("--model", default="outputs/best_model.pt")
    parser.add_argument("--outdir", default="predict_vis")
    args=parser.parse_args()

    import os
    os.makedirs(args.outdir, exist_ok=True)

    # load processor
    processor = AutoProcessor.from_pretrained("facebook/wav2vec2-base")

    # load model
    model = SpeechTransformer(
        pretrained_model_name="facebook/wav2vec2-base",
        mel_feature_dim=80,
        num_speakers=4,    # 你后面我会给你自动推 speaker数版本
        num_digits=10,
    )
    model.load_state_dict(torch.load(args.model,map_location="cpu"))
    model.eval()

    wav, mel = extract_mel(args.audio)
    mel = mel[:128,:]     # 统一长度和训练一样
    if mel.shape[0] < 128:
        mel = torch.cat([mel, torch.zeros(128-mel.shape[0],80)],dim=0)

    inputs = processor(wav.numpy(), sampling_rate=16000, return_tensors="pt", padding=True, return_attention_mask=True)

    speaker_logits, digit_logits, details = model(
        inputs["input_values"],
        inputs["attention_mask"],
        mel.unsqueeze(0),
        return_details=True,
    )

    spk_prob = softmax(speaker_logits[0])
    dig_prob = softmax(digit_logits[0])

    print("=== RESULT ===")
    print("Speaker Prob:", spk_prob.tolist())
    print("Digit Prob:", dig_prob.tolist())

    # save visuals
    plot_and_save_mel(mel, os.path.join(args.outdir,"mel.png"))
    plot_embedding(details["combined_embeddings"][0], os.path.join(args.outdir,"emb.png"))
    plot_attention(details["attention_weights"], os.path.join(args.outdir,"attn.png"))

if __name__=="__main__":
    main()
