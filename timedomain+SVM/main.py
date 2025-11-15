# digit_predictor.py - 使用与训练完全一致的特征提取方法
import os
import librosa
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler


class DigitVoiceRecognizer:
    def __init__(self, model_path='svm_model.pkl', scaler_path='svm_scaler.pkl', verbose=True):
        """初始化数字语音识别器"""
        self.model_path = model_path
        self.scaler_path = scaler_path
        self.model = None
        self.scaler = None
        self.verbose = verbose
        self.load_model()

    def load_model(self):
        """加载训练好的模型和标准化器"""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"模型文件 {self.model_path} 不存在，请先训练模型")
        if not os.path.exists(self.scaler_path):
            raise FileNotFoundError(f"标准化器文件 {self.scaler_path} 不存在")

        with open(self.model_path, 'rb') as f:
            self.model = pickle.load(f)
        with open(self.scaler_path, 'rb') as f:
            self.scaler = pickle.load(f)

        if self.verbose:
            print("模型加载成功！")

    def analyze_single_wav(self, file_path, return_details=False):
        """
        使用与totle.py完全相同的特征提取方法
        返回特征向量，不返回音频数据
        """
        # 1. 读取音频（与训练时相同）
        y, sr = librosa.load(file_path, sr=None, mono=True)
        y = y - np.mean(y)  # 去直流

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

        # 3. 基础时域特征（与训练时完全一致）
        # 短时能量
        ste = np.sum(frames_windowed ** 2, axis=1)
        frame_indices = np.arange(len(ste))
        if len(ste) >= 2:
            ste_slope, _ = np.polyfit(frame_indices, ste, 1)
        else:
            ste_slope = 0

        # 短时平均幅度
        mn = np.sum(np.abs(frames_windowed), axis=1)
        mn_peak = np.max(mn)
        mn_rms = np.sqrt(np.mean(mn ** 2))
        mn_crest = mn_peak / (mn_rms + 1e-6)

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

        # 自相关和AMDF计算函数
        def calc_autocorr(frame, max_lag):
            return np.array([np.sum(frame[:len(frame) - k] * frame[k:]) for k in range(max_lag)])

        def calc_amdf(frame, max_lag):
            return np.array([np.sum(np.abs(frame[:len(frame) - k] - frame[k:])) for k in range(max_lag)])

        # 基频特征提取
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

        # 4. 构建特征向量（与训练时的顺序完全一致）
        features = np.array([
            # 短时能量特征
            np.mean(ste),  # ste_mean
            np.max(ste),  # ste_max
            np.std(ste),  # ste_std
            ste_slope,  # ste_slope

            # 短时平均幅度特征
            np.mean(mn),  # mn_mean
            np.max(mn),  # mn_max
            np.std(mn),  # mn_std
            mn_crest,  # mn_crest

            # 过零率特征
            np.mean(zcr_norm),  # zcr_mean
            np.max(zcr_norm),  # zcr_max
            np.std(zcr_norm),  # zcr_std
            zcr_diff_mean,  # zcr_diff_mean

            # 基频特征
            np.mean(valid_ac) if len(valid_ac) > 0 else 0,  # f0_ac_mean
            np.std(valid_ac) if len(valid_ac) > 0 else 0,  # f0_ac_std
            np.mean(valid_am) if len(valid_am) > 0 else 0,  # f0_am_mean
            np.std(valid_am) if len(valid_am) > 0 else 0,  # f0_am_std
        ])

        if not return_details:
            return features

        feature_vector = {
            "ste_mean": float(features[0]),
            "ste_max": float(features[1]),
            "ste_std": float(features[2]),
            "ste_slope": float(features[3]),
            "mn_mean": float(features[4]),
            "mn_max": float(features[5]),
            "mn_std": float(features[6]),
            "mn_crest": float(features[7]),
            "zcr_mean": float(features[8]),
            "zcr_max": float(features[9]),
            "zcr_std": float(features[10]),
            "zcr_diff_mean": float(features[11]),
            "f0_ac_mean": float(features[12]),
            "f0_ac_std": float(features[13]),
            "f0_am_mean": float(features[14]),
            "f0_am_std": float(features[15])
        }

        duration_seconds = float(len(y) / sr) if sr else 0.0
        feature_summary = {
            "meta": {
                "sample_rate": int(sr),
                "duration_seconds": duration_seconds,
                "frame_count": int(len(ste))
            },
            "groups": [
                {
                    "key": "energy",
                    "title": "短时能量",
                    "metrics": [
                        {"label": "均值", "value": feature_vector["ste_mean"]},
                        {"label": "最大值", "value": feature_vector["ste_max"]},
                        {"label": "标准差", "value": feature_vector["ste_std"]},
                        {"label": "趋势斜率", "value": feature_vector["ste_slope"]}
                    ]
                },
                {
                    "key": "amplitude",
                    "title": "短时平均幅度",
                    "metrics": [
                        {"label": "均值", "value": feature_vector["mn_mean"]},
                        {"label": "最大值", "value": feature_vector["mn_max"]},
                        {"label": "标准差", "value": feature_vector["mn_std"]},
                        {"label": "峰值因子", "value": feature_vector["mn_crest"]}
                    ]
                },
                {
                    "key": "zcr",
                    "title": "过零率",
                    "metrics": [
                        {"label": "均值", "value": feature_vector["zcr_mean"]},
                        {"label": "最大值", "value": feature_vector["zcr_max"]},
                        {"label": "标准差", "value": feature_vector["zcr_std"]},
                        {"label": "差分均值", "value": feature_vector["zcr_diff_mean"]}
                    ]
                },
                {
                    "key": "pitch",
                    "title": "基频估计",
                    "metrics": [
                        {"label": "自相关均值", "value": feature_vector["f0_ac_mean"], "unit": "Hz"},
                        {"label": "自相关标准差", "value": feature_vector["f0_ac_std"], "unit": "Hz"},
                        {"label": "AMDF均值", "value": feature_vector["f0_am_mean"], "unit": "Hz"},
                        {"label": "AMDF标准差", "value": feature_vector["f0_am_std"], "unit": "Hz"}
                    ]
                }
            ],
            "feature_vector": [
                {"key": key, "value": value} for key, value in feature_vector.items()
            ]
        }

        return features, feature_summary

    def predict(self, audio_path):
        """预测音频中的数字"""
        try:
            # 提取特征（使用与训练相同的方法）
            features, feature_summary = self.analyze_single_wav(audio_path, return_details=True)

            # 标准化特征（使用训练时的标准化器）
            features_scaled = self.scaler.transform(features.reshape(1, -1))

            # 预测
            prediction = self.model.predict(features_scaled)[0]
            probabilities = self.model.predict_proba(features_scaled)[0]
            confidence = np.max(probabilities)

            # 获取所有数字的置信度
            digit_confidences = {
                str(i): float(probabilities[i]) for i in range(10)
            }

            return {
                'success': True,
                'digit': int(prediction),
                'confidence': float(confidence),
                'all_confidences': digit_confidences,
                'message': f'识别成功: 数字 {prediction}',
                'time_feature_summary': feature_summary
            }

        except Exception as e:
            return {
                'success': False,
                'digit': -1,
                'confidence': 0.0,
                'all_confidences': {},
                'message': f'识别失败: {str(e)}'
            }

    def predict_batch(self, audio_paths):
        """批量预测多个音频"""
        results = []
        for audio_path in audio_paths:
            result = self.predict(audio_path)
            result['file'] = os.path.basename(audio_path)
            results.append(result)
        return results


# 简单使用示例
def main():
    # 创建识别器
    recognizer = DigitVoiceRecognizer()

    # 单个文件预测
    audio_file = "trimmed.wav"  # 替换为你的音频文件路径
    result = recognizer.predict(audio_file)

    if result['success']:
        print(f"识别结果: 数字 {result['digit']}")
        print(f"置信度: {result['confidence']:.2%}")
        print("所有数字的置信度:")
        for digit, conf in result['all_confidences'].items():
            print(f"  数字 {digit}: {conf:.2%}")
    else:
        print(f"识别失败: {result['message']}")


if __name__ == "__main__":
    main()