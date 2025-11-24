# DSP Assignment Project

这是一个综合性的数字信号处理（DSP）课程作业项目，涵盖了从传统的时域特征提取到现代深度学习模型的多种语音处理技术。项目包含一个交互式的 Web 界面，用于展示和测试各项功能。

## 项目结构

项目主要分为三个部分：

1.  **`dsp-webui/`**: 基于 Web 的交互界面
    *   提供现代化的用户界面，集成语音转文字 (STT)、文字转语音 (TTS) 和大语言模型 (LLM) 对话功能。
    *   采用 Node.js (Web Server) + Python (Chat Service) 的双进程架构。
    *   支持数字识别和声纹锁功能的演示。

2.  **`frequencydomain/`**: 频域分析与深度学习
    *   包含基于 Transformer 的语音识别和声纹识别模型。
    *   提供模型训练、消融实验 (Ablation Studies) 和可视化工具。
    *   实现了智能锁 (Smart Lock) 的声纹验证逻辑。

3.  **`timedomain+SVM/`**: 时域分析与传统机器学习
    *   使用传统的时域特征（短时能量、过零率、基频等）进行数字语音识别。
    *   实现了 SVM, KNN, HMM, Random Forest 等多种分类器。

---

## 1. Web UI (dsp-webui)

这是项目的核心演示界面，整合了后端的所有能力。

### 启动步骤

#### 第一步：安装依赖

**Python 依赖:**
```bash
cd dsp-webui
pip install flask
# 可能还需要安装其他依赖，如 torch, torchaudio, TTS, openai-whisper 等
```

**Node.js 依赖:**
```bash
cd dsp-webui
npm install
```

#### 第二步：启动服务

需要同时运行两个终端窗口。

**终端 1: 启动 Chat Service (Python 后端)**
```bash
cd dsp-webui
python chat_service.py
```
*等待模型加载完成，直到看到 "🚀 Chat Service 已启动" 的提示。*

**终端 2: 启动 Web Server (Node.js 前端)**
```bash
cd dsp-webui
node server.js
```

#### 第三步：访问
打开浏览器访问: [http://localhost:3000](http://localhost:3000)

---

## 2. 频域分析 (frequencydomain)

本模块主要关注使用深度学习模型处理语音信号。

### 主要脚本功能

*   **模型训练**:
    *   `Transformers.py`: 启动 Transformer 模型的训练。
    *   `train_ablate.py`: 进行消融实验 (Full, No Mel, No W2V2, Raw Baseline)。
    *   `train_enhanced.py`: 增强模型的训练。

*   **可视化与预测**:
    *   `visualize.py`: 可视化模型的 Embedding 和 Attention 权重。
    *   `predict_single.py`: 对单个音频文件进行预测。
    *   `plot_ablation.py`: 绘制消融实验对比图。

*   **应用**:
    *   `smart_lock.py`: 智能锁演示，验证说话人身份。

### 示例命令

```bash
cd frequencydomain

# 预测单个音频
python predict_single.py --audio "样本/3/bck-3-3.wav"

# 智能锁验证
python smart_lock.py --audio "lock3.wav" --model "outputs/best_model.pt" --metrics "outputs/metrics.json"
```

---

## 3. 时域分析 (timedomain+SVM)

本模块展示了经典的语音信号处理方法。

### 主要功能

*   **特征提取**: 提取短时能量 (STE)、短时平均幅度 (MN)、过零率 (ZCR)、基频 (Pitch) 等特征。
*   **分类器**:
    *   `SVM.py`: 支持向量机分类器。
    *   `KNN.py`: K-近邻分类器。
    *   `Forest.py`: 随机森林分类器。
    *   `HMM.py`: 隐马尔可夫模型。
*   **数字识别**: `digit_predictor.py` 用于识别录音中的数字 (0-9)。

### 示例

```bash
cd timedomain+SVM
python main.py
```

---

## 环境要求

*   **操作系统**: macOS (推荐), Linux, Windows
*   **Python**: 3.8+
*   **Node.js**: 14+
*   **主要 Python 库**:
    *   `torch`, `torchaudio` (深度学习)
    *   `librosa`, `numpy`, `scipy` (信号处理)
    *   `scikit-learn` (机器学习)
    *   `flask` (Web 服务)
    *   `TTS` (Coqui TTS)
    *   `openai-whisper` (语音识别)

## 注意事项

*   首次启动 `chat_service.py` 时，会自动下载所需的预训练模型（Whisper, XTTS），这可能需要一些时间。
*   请确保在运行脚本前修改脚本中的文件路径为你的本地绝对路径（如果脚本中硬编码了路径）。
