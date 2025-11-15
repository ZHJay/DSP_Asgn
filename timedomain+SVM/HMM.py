import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import librosa
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (accuracy_score, confusion_matrix, classification_report,
                             roc_curve, auc)
from sklearn.preprocessing import label_binarize
import seaborn as sns
from hmmlearn import hmm
import warnings

warnings.filterwarnings('ignore')

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False


# 1. 优化VAD（保留核心功能，避免过度过滤）
def vad(signal, sr, frame_len=2048, hop_len=512):
    energy = np.array(
        [np.sum(np.square(frame)) for frame in librosa.util.frame(signal, frame_length=frame_len, hop_length=hop_len)])
    if len(energy) == 0:
        return signal
    energy_threshold = 0.3 * np.max(energy)  # 用最大值比例，更稳定
    active = energy > energy_threshold
    if np.sum(active) == 0:
        return signal
    start_idx = np.where(active)[0][0] * hop_len
    end_idx = (np.where(active)[0][-1] + 1) * hop_len
    return signal[start_idx:end_idx]


# 2. 简化特征提取（减少维度，避免过拟合）
def extract_features(file_path, sr=16000, n_mfcc=12):
    signal, _ = librosa.load(file_path, sr=sr)
    signal = vad(signal, sr)
    # 过滤过短语音（至少0.2秒，确保有足够帧数）
    if len(signal) < 0.2 * sr:
        return None
    # 只提取MFCC和一阶差分（24维，减少复杂度）
    mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=n_mfcc, n_fft=2048, hop_length=512)
    mfcc_delta = librosa.feature.delta(mfcc, width=3)  # 仅一阶差分
    features = np.concatenate([mfcc, mfcc_delta], axis=0).T  # [帧数, 24]
    # 特征归一化（避免数值异常）
    scaler = MinMaxScaler(feature_range=(0.01, 0.99))  # 避免0和1导致概率溢出
    return scaler.fit_transform(features)


# 3. 构建数据集（严格过滤无效样本）
def build_dataset(audio_dir='D:\sample'):
    X, y = [], []
    for digit in range(10):
        digit_dir = os.path.join(audio_dir, str(digit))
        if not os.path.exists(digit_dir):
            print(f"警告：未找到数字{digit}的音频文件夹")
            continue
        for file in os.listdir(digit_dir):
            if file.endswith('.wav'):
                feat = extract_features(os.path.join(digit_dir, file))
                # 确保特征帧数≥8（足够HMM学习时序）
                if feat is not None and len(feat) >= 8:
                    X.append(feat)
                    y.append(digit)
    print(f"成功构建数据集：{len(X)}个样本，涵盖数字{set(y)}")
    # 检查类别分布，确保每个数字至少有5个样本
    for digit in range(10):
        cnt = y.count(digit)
        if cnt < 5:
            print(f"警告：数字{digit}仅{cnt}个样本，可能影响模型性能")
    return X, y


# 4. 简化HMM模型（适配小样本）
class DigitHMM:
    def __init__(self, n_states=2, max_iter=20):
        self.n_states = n_states
        self.max_iter = max_iter
        self.models = {}
        self.scaler = MinMaxScaler(feature_range=(0.01, 0.99))

    def train(self, X_train, y_train):
        # 全局特征归一化
        all_feats = np.concatenate(X_train, axis=0)
        self.scaler.fit(all_feats)
        n_dim = all_feats.shape[1]  # 特征维度（24维）

        for digit in np.unique(y_train):
            # 提取并增强数据
            digit_feats = [self.scaler.transform(feat) for i, feat in enumerate(X_train) if y_train[i] == digit]
            while len(digit_feats) < 10:
                digit_feats.extend([f + np.random.normal(0, 0.01, f.shape) for f in digit_feats])
            lengths = [len(feat) for feat in digit_feats]
            X_digit = np.concatenate(digit_feats, axis=0)

            # 核心修复：添加covars_prior，训练时直接生成非零协方差
            model = hmm.GaussianHMM(
                n_components=self.n_states,
                covariance_type='diag',
                transmat_prior=0.2,
                covars_prior=0.01,  # 协方差先验，避免训练后协方差为0
                init_params='cw',  # 仅初始化协方差和发射概率
                n_iter=self.max_iter,
                tol=1e-3,
                random_state=42
            )

            # 手动设置初始概率和转移矩阵
            model.startprob_ = np.array([0.8, 0.2])
            transmat = np.array([[0.7, 0.3], [0.0, 1.0]])
            model.transmat_ = transmat

            # 移除训练后修改covars_的代码，无需再手动调整
            model.fit(X_digit, lengths)
            self.models[digit] = model
            print(f"数字{digit}的HMM训练完成（迭代{model.monitor_.iter}次）")

    def predict(self, X_test):
        y_pred = []
        y_proba = []
        for feat in X_test:
            feat_scaled = self.scaler.transform(feat)
            log_probs = {}
            for digit in range(10):
                if digit not in self.models:
                    log_probs[digit] = -np.inf
                    continue
                try:
                    log_prob = self.models[digit].score(feat_scaled)
                    log_prob = np.clip(log_prob, -1e5, 1e5)
                    log_probs[digit] = log_prob
                except:
                    log_probs[digit] = -1e5

            # 处理概率异常值
            probs = np.exp([log_probs[d] for d in range(10)])
            probs[np.isnan(probs)] = 1e-10
            probs[probs < 1e-10] = 1e-10
            probs /= np.sum(probs)
            y_proba.append(probs)
            y_pred.append(np.argmax(probs))
        return np.array(y_pred), np.array(y_proba)


# 5. 主函数（确保AUC曲线正常生成）
def main():
    # 加载数据
    X, y = build_dataset()
    if len(X) < 50:
        print("样本数过少，无法训练模型")
        return

    # 划分数据集（确保每类至少有2个测试样本）
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    print(f"训练集：{len(X_train)}个样本，测试集：{len(X_test)}个样本")

    # 训练模型
    hmm_model = DigitHMM(n_states=2)
    hmm_model.train(X_train, y_train)

    # 预测与评估
    y_pred, y_proba = hmm_model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\n测试集准确率：{acc:.2%}")

    # 输出评估指标
    print("\n===== 分类器详细评估 =====")
    print(f"准确率：{acc:.2%}")
    cm = confusion_matrix(y_test, y_pred)
    print("\n混淆矩阵：")
    print(cm)
    print("\n分类报告：")
    # 处理无预测样本的类别，避免警告
    target_names = [str(i) for i in range(10)]
    print(classification_report(y_test, y_pred, target_names=target_names, zero_division=0))

    # 可视化混淆矩阵
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens',
                xticklabels=target_names, yticklabels=target_names)
    plt.xlabel('预测标签')
    plt.ylabel('真实标签')
    plt.title('HMM分类器混淆矩阵')
    plt.tight_layout()
    plt.savefig('hmm_confusion_matrix.png', dpi=300)
    plt.show()

    # 计算并绘制AUC曲线（修复NaN问题）
    y_test_bin = label_binarize(y_test, classes=range(10))
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for i in range(10):
        # 过滤掉无真实样本的类别
        if np.sum(y_test_bin[:, i]) == 0:
            fpr[i] = [0.0, 1.0]
            tpr[i] = [0.0, 1.0]
            roc_auc[i] = 0.5
            continue
        # 确保概率无NaN
        y_score = y_proba[:, i].copy()
        y_score[np.isnan(y_score)] = 0.0
        # 计算ROC曲线
        fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score)
        roc_auc[i] = auc(fpr[i], tpr[i])

    # 宏平均AUC
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(10)]))
    mean_tpr = np.zeros_like(all_fpr)
    valid_classes = 0
    for i in range(10):
        if roc_auc[i] != 0.5:  # 只计算有有效样本的类别
            mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
            valid_classes += 1
    if valid_classes > 0:
        mean_tpr /= valid_classes
    roc_auc_macro = auc(all_fpr, mean_tpr)

    # 绘制AUC曲线（与原样式一致）
    plt.figure(figsize=(6, 6))
    plt.plot(all_fpr, mean_tpr, 'green', linewidth=2, label='ROC曲线')
    plt.fill_between(all_fpr, mean_tpr, alpha=0.2, color='green', label='AUC区域')
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1.5)
    # 选择有效的工作点
    work_point_idx = min(1, len(all_fpr) - 1)
    plt.scatter(all_fpr[work_point_idx], mean_tpr[work_point_idx], color='r', s=60, label='工作点')
    plt.text(all_fpr[work_point_idx] + 0.03, mean_tpr[work_point_idx] - 0.08,
             f'({all_fpr[work_point_idx]:.2f},{mean_tpr[work_point_idx]:.2f})', color='r', fontsize=10)
    plt.text(0.4, 0.3, f'AUC = {roc_auc_macro:.3f}', fontsize=12, fontweight='bold')
    plt.xlabel('假正率')
    plt.ylabel('真正率')
    plt.title('HMM模型')
    plt.legend(loc='lower right', frameon=True, facecolor='white', edgecolor='gray')
    plt.grid(linestyle='--', alpha=0.3)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.savefig('hmm_auc.png', dpi=300, bbox_inches='tight')
    plt.show()
    print(f"宏平均AUC值：{roc_auc_macro:.3f}")


if __name__ == "__main__":
    main()