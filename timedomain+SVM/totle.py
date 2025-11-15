import os
import librosa
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False


def analyze_single_wav(file_path, digit):
    """分析单个wav文件的时域特征"""
    # 1. 读取音频
    y, sr = librosa.load(file_path, sr=None, mono=True)
    y = y - np.mean(y)  # 去直流
    duration = len(y) / sr  # 音频时长（秒）

    # 2. 分帧参数（48000Hz下，20ms帧长+10ms帧移）
    framelen = 960
    inc = 480
    nx = len(y)
    num_frames = int(np.ceil((nx - framelen + inc) / inc))
    y_padded = np.pad(y, (0, (num_frames - 1) * inc + framelen - nx), mode='constant')
    frames = np.array([y_padded[i * inc:i * inc + framelen] for i in range(num_frames)])
    # 加窗
    window = np.hanning(framelen)
    frames_windowed = frames * window
    frame_time = (np.arange(num_frames) * inc + framelen / 2) / sr

    # 3. 基础时域特征
    # 短时能量
    ste = np.sum(frames_windowed **2, axis=1)
    frame_indices = np.arange(len(ste))  # 帧索引
    if len(ste) >= 2:
        ste_slope, _ = np.polyfit(frame_indices, ste, 1)  # 一阶线性拟合
    else:
        ste_slope = 0  # 单帧时斜率为0
    ste_stats = {
        'ste_mean': np.mean(ste),
        'ste_max': np.max(ste),
        'ste_std': np.std(ste),
        'ste_slope':ste_slope
    }

    # 短时平均幅度
    mn = np.sum(np.abs(frames_windowed), axis=1)
    mn_peak = np.max(mn)
    mn_rms = np.sqrt(np.mean(mn ** 2))
    mn_crest = mn_peak / (mn_rms + 1e-6)
    mn_stats = {
        'mn_mean': np.mean(mn),
        'mn_max': np.max(mn),
        'mn_std': np.std(mn),
        'mn_crest':mn_crest
    }

    # 短时过零率
    def calc_zcr(frame):
        sign_diff = np.abs(np.sign(frame[1:]) - np.sign(frame[:-1]))
        return np.sum(sign_diff) / 2
    zcr = np.array([calc_zcr(frame) for frame in frames_windowed])
    zcr_norm = zcr / framelen
    if len(zcr_norm) >= 2:
        zcr_diff = np.diff(zcr_norm)
        zcr_diff_mean = np.mean(np.abs(zcr_diff))
    else:
        zcr_diff_mean = 0
    zcr_stats = {
        'zcr_mean': np.mean(zcr_norm),
        'zcr_max': np.max(zcr_norm),
        'zcr_std': np.std(zcr_norm),
        'zcr_diff_mean': zcr_diff_mean
    }

    #自相关和AMDF计算函数
    def calc_autocorr(frame, max_lag):
        return np.array([np.sum(frame[:len(frame) - k] * frame[k:]) for k in range(max_lag)])

    def calc_amdf(frame, max_lag):
        return np.array([np.sum(np.abs(frame[:len(frame) - k] - frame[k:])) for k in range(max_lag)])

    min_f0, max_f0 = 80, 400
    min_lag = int(sr / max_f0)
    max_lag = int(sr / min_f0)
    f0_ac_all = []
    f0_am_all = []
    for i, frame in enumerate(frames_windowed):
        frame_len = len(frame)
        current_max_lag = min(max_lag, frame_len // 2)

        # 无效帧基频设为0
        if current_max_lag <= min_lag or ste[i] < 0.05 * np.max(ste):
            f0_ac_all.append(0)
            f0_am_all.append(0)
            continue
        autocorr = calc_autocorr(frame, current_max_lag)
        amdf = calc_amdf(frame, current_max_lag)
        # 自相关法基频（AC峰值）
        ac_peak_idx = min_lag + np.argmax(autocorr[min_lag:current_max_lag])
        f0_ac = sr / ac_peak_idx if ac_peak_idx != 0 else 0
        f0_ac_all.append(f0_ac)

        # AMDF法基频（AMDF谷值）
        am_valley_idx = min_lag + np.argmin(amdf[min_lag:current_max_lag])
        f0_am = sr / am_valley_idx if am_valley_idx != 0 else 0
        f0_am_all.append(f0_am)

    f0_ac_array = np.array(f0_ac_all)
    f0_am_array = np.array(f0_am_all)
    valid_ac = f0_ac_array[(f0_ac_array >= min_f0) & (f0_ac_array <= max_f0)]
    valid_am = f0_am_array[(f0_am_array >= min_f0) & (f0_am_array <= max_f0)]
    pitch_stats = {
        # 自相关法基频特征
        'f0_ac_mean': np.mean(valid_ac) if len(valid_ac) > 0 else 0,
        'f0_ac_std': np.std(valid_ac) if len(valid_ac) > 0 else 0,
        # AMDF法基频特征
        'f0_am_mean': np.mean(valid_am) if len(valid_am) > 0 else 0,
        'f0_am_std': np.std(valid_am) if len(valid_am) > 0 else 0,

    }
    # 5. 整合所有特征
    features = {
        'file_name': os.path.basename(file_path),
        'digit': digit,
        ** ste_stats,
        **mn_stats,** zcr_stats,
        **pitch_stats
    }
    return features, y


def batch_analyze(root_dir, save_path='digit_features.csv'):
    """批量分析所有数字0-9的所有音频文件"""
    all_features = []
    for digit in range(10):
        digit_dir = os.path.join(root_dir, str(digit))
        if not os.path.exists(digit_dir):
            print(f"警告：文件夹 {digit_dir} 不存在，跳过")
            continue
        wav_files = [f for f in os.listdir(digit_dir) if f.endswith('.wav')]
        print(f"开始处理数字 {digit}，共 {len(wav_files)} 个文件")
        for file in wav_files:
            file_path = os.path.join(digit_dir, file)
            try:
                features, _ = analyze_single_wav(file_path, digit)
                all_features.append(features)
            except Exception as e:
                print(f"处理 {file} 失败：{e}")
                continue

    df = pd.DataFrame(all_features)
    df.to_csv(save_path, index=False)
    print(f"所有文件处理完成，特征已保存到 {save_path}")
    return df


def analyze_features(csv_path='digit_features.csv'):
    df = pd.read_csv(csv_path)
    digit_stats = df.groupby('digit').mean(numeric_only=True)
    return digit_stats


# 主函数
def main():
    root_dir = "D:\\sample"  # 根目录下有0-9文件夹

    print("=== 开始批量分析0-9所有数字的特征 ===")

    # 1. 批量分析特征（处理所有数字的所有文件）
    df = batch_analyze(root_dir)

    # 2. 分析特征统计数据
    digit_stats = analyze_features()


# 运行主函数
if __name__ == "__main__":
    main()