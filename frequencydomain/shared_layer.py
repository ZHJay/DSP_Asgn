"""
共享隐空间架构 - 同时支持STT和TTS
任务1: 语音 → 数字分类 + 说话人识别
任务2: 文本+说话人 → 语音生成 (在TTS.py中使用)

设计理念：
- 音频和文本编码到同一隐空间
- 使用多任务学习同时优化分类和生成
- 对比学习确保音频和文本对齐
"""

import os
import re
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import librosa
from tqdm import tqdm

# =============================
# 配置
# =============================
class Config:
    # 数据路径
    SAMPLE_DIR = r"D:\DSP\STT+TTS\样本"
    MODEL_SAVE_DIR = r"D:\DSP\STT+TTS\models"
    
    # 音频参数
    SR = 16000
    N_FFT = 512
    HOP_LENGTH = 160  # 10ms
    WIN_LENGTH = 400  # 25ms
    N_MELS = 80
    MAX_FRAMES = 200
    
    # 模型参数
    LATENT_DIM = 256        # 共享隐空间维度
    AUDIO_ENCODER_DIM = 128
    TEXT_ENCODER_DIM = 128
    N_HEADS = 8
    N_LAYERS = 4
    DROPOUT = 0.1
    
    # 训练参数
    BATCH_SIZE = 8
    EPOCHS = 200
    LR = 1e-4
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 任务参数
    N_DIGITS = 10  # 0-9
    N_SPEAKERS = 4  # bck, cqc, xsq, zhj
    
    # 说话人映射
    SPEAKER_TO_ID = {
        'bck': 0,
        'cqc': 1,
        'xsq': 2,
        'zhj': 3
    }
    ID_TO_SPEAKER = {v: k for k, v in SPEAKER_TO_ID.items()}

config = Config()

# =============================
# 数据预处理和加载
# =============================

def parse_filename(filename):
    """
    解析文件名: 姓名-数字-编号.wav
    例如: bck-3-5.wav -> speaker='bck', digit=3, idx=5
    """
    match = re.match(r"([a-z]+)-(\d+)-(\d+)\.wav$", filename)
    if match:
        speaker = match.group(1)
        digit = int(match.group(2))
        idx = int(match.group(3))
        return speaker, digit, idx
    else:
        raise ValueError(f"文件名格式错误: {filename}")

def extract_mel_spectrogram(y, sr=16000):
    """提取Mel频谱图"""
    mel_spec = librosa.feature.melspectrogram(
        y=y, 
        sr=sr,
        n_fft=config.N_FFT,
        hop_length=config.HOP_LENGTH,
        win_length=config.WIN_LENGTH,
        n_mels=config.N_MELS
    )
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    return mel_spec_db.T  # (frames, n_mels)

def preprocess_audio(y, sr):
    """音频预处理"""
    # 去均值
    y = y - np.mean(y)
    
    # 归一化
    max_val = np.abs(y).max()
    if max_val > 0:
        y = y / max_val
    
    # 预加重
    y = np.append(y[0], y[1:] - 0.97 * y[:-1])
    
    # 去静音
    y, _ = librosa.effects.trim(y, top_db=20)
    
    return y

def load_and_process_audio(file_path, max_frames=200):
    """加载并处理音频文件"""
    # 加载
    y, sr = librosa.load(file_path, sr=config.SR)
    
    # 预处理
    y = preprocess_audio(y, sr)
    
    # 提取Mel频谱
    mel_spec = extract_mel_spectrogram(y, sr)
    
    # 对齐长度
    if mel_spec.shape[0] > max_frames:
        mel_spec = mel_spec[:max_frames, :]
    else:
        pad_len = max_frames - mel_spec.shape[0]
        mel_spec = np.vstack([mel_spec, np.zeros((pad_len, config.N_MELS))])
    
    return mel_spec

class AudioTextDataset(Dataset):
    """音频-文本配对数据集"""
    def __init__(self, data_dir, split='train'):
        self.data_dir = data_dir
        self.samples = []
        
        # 扫描所有文件
        for digit_folder in sorted(os.listdir(data_dir)):
            folder_path = os.path.join(data_dir, digit_folder)
            if not os.path.isdir(folder_path):
                continue
            
            try:
                digit = int(digit_folder)
            except ValueError:
                continue
            
            for wav_file in os.listdir(folder_path):
                if wav_file.endswith('.wav'):
                    try:
                        speaker, digit_parsed, idx = parse_filename(wav_file)
                        if speaker in config.SPEAKER_TO_ID:
                            self.samples.append({
                                'path': os.path.join(folder_path, wav_file),
                                'speaker': speaker,
                                'speaker_id': config.SPEAKER_TO_ID[speaker],
                                'digit': digit_parsed,
                                'filename': wav_file
                            })
                    except ValueError:
                        continue
        
        print(f"加载了 {len(self.samples)} 个样本")
        print(f"说话人: {list(config.SPEAKER_TO_ID.keys())}")
        print(f"数字: 0-9")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # 加载音频
        mel_spec = load_and_process_audio(sample['path'], config.MAX_FRAMES)
        
        return {
            'mel_spec': torch.FloatTensor(mel_spec),  # (frames, n_mels)
            'digit': torch.LongTensor([sample['digit']]),
            'speaker_id': torch.LongTensor([sample['speaker_id']]),
            'speaker': sample['speaker'],
            'text': f"{sample['speaker']}_{sample['digit']}"  # 用于对比学习
        }

# =============================
# 共享隐空间模型架构
# =============================

class PositionalEncoding(nn.Module):
    """位置编码"""
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))
    
    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

class AudioEncoder(nn.Module):
    """音频编码器: Mel Spectrogram → Latent Space"""
    def __init__(self, n_mels=80, latent_dim=256, d_model=128, n_heads=8, n_layers=4, dropout=0.1):
        super().__init__()
        
        # 输入投影
        self.input_proj = nn.Sequential(
            nn.Linear(n_mels, d_model),
            nn.LayerNorm(d_model),
            nn.Dropout(dropout)
        )
        
        # 位置编码
        self.pos_encoding = PositionalEncoding(d_model)
        
        # Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        # 池化到latent space
        self.attention_pool = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.Tanh(),
            nn.Linear(d_model // 2, 1)
        )
        
        # 投影到latent space
        self.to_latent = nn.Sequential(
            nn.Linear(d_model, latent_dim),
            nn.LayerNorm(latent_dim)
        )
    
    def forward(self, mel_spec):
        """
        mel_spec: (batch, frames, n_mels)
        返回: (batch, latent_dim)
        """
        # 输入投影
        x = self.input_proj(mel_spec)  # (B, T, d_model)
        
        # 位置编码
        x = self.pos_encoding(x)
        
        # Transformer编码
        x = self.transformer(x)  # (B, T, d_model)
        
        # 注意力池化
        attn_weights = torch.softmax(self.attention_pool(x).squeeze(-1), dim=1)  # (B, T)
        pooled = torch.bmm(attn_weights.unsqueeze(1), x).squeeze(1)  # (B, d_model)
        
        # 投影到latent space
        latent = self.to_latent(pooled)  # (B, latent_dim)
        
        return latent, attn_weights

class TextEncoder(nn.Module):
    """文本编码器: (Speaker ID + Digit) → Latent Space"""
    def __init__(self, n_speakers=4, n_digits=10, latent_dim=256, d_model=128):
        super().__init__()
        
        # Embedding
        self.speaker_embed = nn.Embedding(n_speakers, d_model // 2)
        self.digit_embed = nn.Embedding(n_digits, d_model // 2)
        
        # 融合和投影
        self.fusion = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_model, latent_dim),
            nn.LayerNorm(latent_dim)
        )
    
    def forward(self, speaker_id, digit):
        """
        speaker_id: (batch,)
        digit: (batch,)
        返回: (batch, latent_dim)
        """
        speaker_emb = self.speaker_embed(speaker_id)  # (B, d_model//2)
        digit_emb = self.digit_embed(digit)  # (B, d_model//2)
        
        # 拼接
        combined = torch.cat([speaker_emb, digit_emb], dim=-1)  # (B, d_model)
        
        # 投影到latent space
        latent = self.fusion(combined)  # (B, latent_dim)
        
        return latent

class AudioDecoder(nn.Module):
    """音频解码器: Latent Space → Mel Spectrogram (用于TTS)"""
    def __init__(self, latent_dim=256, n_mels=80, max_frames=200, d_model=128, n_heads=8, n_layers=4, dropout=0.1):
        super().__init__()
        self.max_frames = max_frames
        
        # Latent到序列
        self.latent_to_seq = nn.Sequential(
            nn.Linear(latent_dim, d_model * max_frames),
            nn.LayerNorm(d_model * max_frames),
            nn.GELU()
        )
        
        # 位置编码
        self.pos_encoding = PositionalEncoding(d_model, max_frames)
        
        # Transformer解码器
        decoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(decoder_layer, num_layers=n_layers)
        
        # 输出投影
        self.output_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, n_mels)
        )
    
    def forward(self, latent):
        """
        latent: (batch, latent_dim)
        返回: (batch, max_frames, n_mels)
        """
        batch_size = latent.size(0)
        
        # 展开为序列
        x = self.latent_to_seq(latent)  # (B, d_model * max_frames)
        x = x.view(batch_size, self.max_frames, -1)  # (B, max_frames, d_model)
        
        # 位置编码
        x = self.pos_encoding(x)
        
        # Transformer解码
        x = self.transformer(x)  # (B, max_frames, d_model)
        
        # 输出投影
        mel_spec = self.output_proj(x)  # (B, max_frames, n_mels)
        
        return mel_spec

class SharedLatentModel(nn.Module):
    """
    共享隐空间模型
    - AudioEncoder: 音频 → latent
    - TextEncoder: 文本 → latent
    - AudioDecoder: latent → 音频
    - Classifiers: latent → 数字/说话人
    """
    def __init__(self, config):
        super().__init__()
        
        self.audio_encoder = AudioEncoder(
            n_mels=config.N_MELS,
            latent_dim=config.LATENT_DIM,
            d_model=config.AUDIO_ENCODER_DIM,
            n_heads=config.N_HEADS,
            n_layers=config.N_LAYERS,
            dropout=config.DROPOUT
        )
        
        self.text_encoder = TextEncoder(
            n_speakers=config.N_SPEAKERS,
            n_digits=config.N_DIGITS,
            latent_dim=config.LATENT_DIM,
            d_model=config.TEXT_ENCODER_DIM
        )
        
        self.audio_decoder = AudioDecoder(
            latent_dim=config.LATENT_DIM,
            n_mels=config.N_MELS,
            max_frames=config.MAX_FRAMES,
            d_model=config.AUDIO_ENCODER_DIM,
            n_heads=config.N_HEADS,
            n_layers=config.N_LAYERS,
            dropout=config.DROPOUT
        )
        
        # 分类头
        self.digit_classifier = nn.Sequential(
            nn.Linear(config.LATENT_DIM, config.LATENT_DIM // 2),
            nn.LayerNorm(config.LATENT_DIM // 2),
            nn.GELU(),
            nn.Dropout(config.DROPOUT),
            nn.Linear(config.LATENT_DIM // 2, config.N_DIGITS)
        )
        
        self.speaker_classifier = nn.Sequential(
            nn.Linear(config.LATENT_DIM, config.LATENT_DIM // 2),
            nn.LayerNorm(config.LATENT_DIM // 2),
            nn.GELU(),
            nn.Dropout(config.DROPOUT),
            nn.Linear(config.LATENT_DIM // 2, config.N_SPEAKERS)
        )
    
    def forward(self, mel_spec=None, speaker_id=None, digit=None, mode='audio'):
        """
        mode='audio': 音频输入，返回分类结果
        mode='text': 文本输入，返回latent
        mode='generate': 文本输入，返回生成的音频
        """
        if mode == 'audio':
            # STT模式: 音频 → latent → 分类
            audio_latent, attn_weights = self.audio_encoder(mel_spec)
            digit_logits = self.digit_classifier(audio_latent)
            speaker_logits = self.speaker_classifier(audio_latent)
            return {
                'latent': audio_latent,
                'digit_logits': digit_logits,
                'speaker_logits': speaker_logits,
                'attention_weights': attn_weights
            }
        
        elif mode == 'text':
            # 文本编码模式
            text_latent = self.text_encoder(speaker_id, digit)
            return {
                'latent': text_latent
            }
        
        elif mode == 'generate':
            # TTS模式: 文本 → latent → 音频
            text_latent = self.text_encoder(speaker_id, digit)
            generated_mel = self.audio_decoder(text_latent)
            return {
                'latent': text_latent,
                'mel_spec': generated_mel
            }
        
        elif mode == 'full':
            # 完整训练模式: 音频→latent, 文本→latent, latent→音频, latent→分类
            audio_latent, attn_weights = self.audio_encoder(mel_spec)
            text_latent = self.text_encoder(speaker_id, digit)
            reconstructed_mel = self.audio_decoder(audio_latent)
            digit_logits = self.digit_classifier(audio_latent)
            speaker_logits = self.speaker_classifier(audio_latent)
            
            return {
                'audio_latent': audio_latent,
                'text_latent': text_latent,
                'reconstructed_mel': reconstructed_mel,
                'digit_logits': digit_logits,
                'speaker_logits': speaker_logits,
                'attention_weights': attn_weights
            }

# =============================
# 损失函数
# =============================

class MultiTaskLoss(nn.Module):
    """多任务损失"""
    def __init__(self, alpha_cls=1.0, alpha_recon=1.0, alpha_contrast=0.5):
        super().__init__()
        self.alpha_cls = alpha_cls
        self.alpha_recon = alpha_recon
        self.alpha_contrast = alpha_contrast
        
        self.ce_loss = nn.CrossEntropyLoss()
        self.mse_loss = nn.MSELoss()
    
    def contrastive_loss(self, audio_latent, text_latent, temperature=0.07):
        """对比学习损失 - 让音频和文本的latent对齐"""
        # L2归一化
        audio_latent = F.normalize(audio_latent, dim=1)
        text_latent = F.normalize(text_latent, dim=1)
        
        # 计算相似度矩阵
        logits = torch.matmul(audio_latent, text_latent.T) / temperature
        
        # 标签是对角线
        batch_size = audio_latent.size(0)
        labels = torch.arange(batch_size, device=audio_latent.device)
        
        # 双向对比损失
        loss_a2t = self.ce_loss(logits, labels)
        loss_t2a = self.ce_loss(logits.T, labels)
        
        return (loss_a2t + loss_t2a) / 2
    
    def forward(self, outputs, targets):
        """
        outputs: 模型输出字典
        targets: 目标字典 {'digit', 'speaker_id', 'mel_spec'}
        """
        losses = {}
        
        # 分类损失
        if 'digit_logits' in outputs:
            losses['digit_loss'] = self.ce_loss(
                outputs['digit_logits'], 
                targets['digit'].squeeze(1)
            )
        
        if 'speaker_logits' in outputs:
            losses['speaker_loss'] = self.ce_loss(
                outputs['speaker_logits'],
                targets['speaker_id'].squeeze(1)
            )
        
        # 重建损失
        if 'reconstructed_mel' in outputs:
            losses['recon_loss'] = self.mse_loss(
                outputs['reconstructed_mel'],
                targets['mel_spec']
            )
        
        # 对比学习损失
        if 'audio_latent' in outputs and 'text_latent' in outputs:
            losses['contrast_loss'] = self.contrastive_loss(
                outputs['audio_latent'],
                outputs['text_latent']
            )
        
        # 总损失
        total_loss = 0
        if 'digit_loss' in losses:
            total_loss += self.alpha_cls * losses['digit_loss']
        if 'speaker_loss' in losses:
            total_loss += self.alpha_cls * losses['speaker_loss']
        if 'recon_loss' in losses:
            total_loss += self.alpha_recon * losses['recon_loss']
        if 'contrast_loss' in losses:
            total_loss += self.alpha_contrast * losses['contrast_loss']
        
        losses['total_loss'] = total_loss
        
        return losses

# =============================
# 训练函数
# =============================

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """训练一个epoch"""
    model.train()
    total_losses = {
        'total': 0,
        'digit': 0,
        'speaker': 0,
        'recon': 0,
        'contrast': 0
    }
    digit_correct = 0
    speaker_correct = 0
    total_samples = 0
    
    pbar = tqdm(dataloader, desc="Training")
    for batch in pbar:
        mel_spec = batch['mel_spec'].to(device)
        digit = batch['digit'].to(device)
        speaker_id = batch['speaker_id'].to(device)
        
        # 前向传播
        outputs = model(
            mel_spec=mel_spec,
            speaker_id=speaker_id.squeeze(1),
            digit=digit.squeeze(1),
            mode='full'
        )
        
        # 计算损失
        targets = {
            'mel_spec': mel_spec,
            'digit': digit,
            'speaker_id': speaker_id
        }
        losses = criterion(outputs, targets)
        
        # 反向传播
        optimizer.zero_grad()
        losses['total_loss'].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        # 统计
        total_losses['total'] += losses['total_loss'].item()
        total_losses['digit'] += losses.get('digit_loss', torch.tensor(0)).item()
        total_losses['speaker'] += losses.get('speaker_loss', torch.tensor(0)).item()
        total_losses['recon'] += losses.get('recon_loss', torch.tensor(0)).item()
        total_losses['contrast'] += losses.get('contrast_loss', torch.tensor(0)).item()
        
        digit_correct += (outputs['digit_logits'].argmax(1) == digit.squeeze(1)).sum().item()
        speaker_correct += (outputs['speaker_logits'].argmax(1) == speaker_id.squeeze(1)).sum().item()
        total_samples += mel_spec.size(0)
        
        pbar.set_postfix({
            'loss': f"{losses['total_loss'].item():.4f}",
            'digit_acc': f"{digit_correct/total_samples:.3f}",
            'spk_acc': f"{speaker_correct/total_samples:.3f}"
        })
    
    return {
        'loss': total_losses['total'] / len(dataloader),
        'digit_acc': digit_correct / total_samples,
        'speaker_acc': speaker_correct / total_samples
    }

def evaluate(model, dataloader, criterion, device):
    """评估"""
    model.eval()
    total_losses = {'total': 0, 'digit': 0, 'speaker': 0}
    digit_correct = 0
    speaker_correct = 0
    total_samples = 0
    
    with torch.no_grad():
        for batch in dataloader:
            mel_spec = batch['mel_spec'].to(device)
            digit = batch['digit'].to(device)
            speaker_id = batch['speaker_id'].to(device)
            
            outputs = model(mel_spec=mel_spec, mode='audio')
            
            targets = {'digit': digit, 'speaker_id': speaker_id, 'mel_spec': mel_spec}
            losses = criterion(outputs, targets)
            
            total_losses['total'] += losses['total_loss'].item()
            digit_correct += (outputs['digit_logits'].argmax(1) == digit.squeeze(1)).sum().item()
            speaker_correct += (outputs['speaker_logits'].argmax(1) == speaker_id.squeeze(1)).sum().item()
            total_samples += mel_spec.size(0)
    
    return {
        'loss': total_losses['total'] / len(dataloader),
        'digit_acc': digit_correct / total_samples,
        'speaker_acc': speaker_correct / total_samples
    }

# =============================
# 主训练循环
# =============================

def train_model():
    """训练共享隐空间模型"""
    print("=" * 60)
    print("开始训练共享隐空间模型")
    print("=" * 60)
    
    # 创建保存目录
    os.makedirs(config.MODEL_SAVE_DIR, exist_ok=True)
    
    # 加载数据
    dataset = AudioTextDataset(config.SAMPLE_DIR)
    
    # 划分训练集和验证集
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=0
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )
    
    # 创建模型
    model = SharedLatentModel(config).to(config.DEVICE)
    print(f"\n模型参数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 损失函数和优化器
    criterion = MultiTaskLoss(alpha_cls=1.0, alpha_recon=1.0, alpha_contrast=0.5)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.EPOCHS)
    
    # 训练
    best_val_acc = 0
    for epoch in range(config.EPOCHS):
        print(f"\nEpoch {epoch+1}/{config.EPOCHS}")
        
        train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, config.DEVICE)
        val_metrics = evaluate(model, val_loader, criterion, config.DEVICE)
        
        scheduler.step()
        
        print(f"Train - Loss: {train_metrics['loss']:.4f}, "
              f"Digit Acc: {train_metrics['digit_acc']:.4f}, "
              f"Speaker Acc: {train_metrics['speaker_acc']:.4f}")
        print(f"Val   - Loss: {val_metrics['loss']:.4f}, "
              f"Digit Acc: {val_metrics['digit_acc']:.4f}, "
              f"Speaker Acc: {val_metrics['speaker_acc']:.4f}")
        
        # 保存最佳模型
        avg_val_acc = (val_metrics['digit_acc'] + val_metrics['speaker_acc']) / 2
        if avg_val_acc > best_val_acc:
            best_val_acc = avg_val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_digit_acc': val_metrics['digit_acc'],
                'val_speaker_acc': val_metrics['speaker_acc'],
                'config': config
            }, os.path.join(config.MODEL_SAVE_DIR, 'best_model.pth'))
            print(f"✅ 保存最佳模型! 平均准确率: {avg_val_acc:.4f}")
    
    print("\n训练完成!")
    return model

if __name__ == "__main__":
    print(f"Device: {config.DEVICE}")
    model = train_model()

