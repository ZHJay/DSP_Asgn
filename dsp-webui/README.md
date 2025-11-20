# 音频处理控制台

一个集成了数字识别和声音克隆功能的 Web 应用。

## 功能特点

### 数字识别
- **文件上传与预处理**：上传 WAV 音频文件进行预处理
- **时域分析**：基于时域特征的数字识别
- **频域分析**：基于频域特征的数字和说话人识别
- **可视化**：概率分布图表和特征摘要

### 声音克隆
- **零样本语音合成**：使用 XTTS-v2 模型进行声音克隆
- **多语言支持**：支持中文和英文合成
- **参考音频克隆**：上传参考音频，生成相同音色的新语音
- **实时播放与下载**：在线播放合成结果，支持下载

## 使用方法

### 启动服务器

```bash
cd dsp-webui
node server.js
```

服务器将在 http://localhost:3000 启动

### 数字识别流程

1. 切换到"数字识别"标签
2. 选择 WAV 音频文件并上传
3. 等待预处理完成
4. 点击"时域分析"或"频域分析"按钮
5. 查看识别结果和概率分布

### 声音克隆流程

1. 切换到"声音克隆"标签
2. 选择参考音频文件（WAV 格式）
3. 输入要合成的文本内容
4. 选择语言（中文或英文）
5. 点击"开始合成"
6. 等待合成完成后播放或下载

## 技术栈

### 前端
- Vue.js 3
- 响应式 CSS
- HTML5 Audio API

### 后端
- Node.js + Express
- Multer (文件上传)
- Child Process (Python 脚本调用)

### AI 模型
- **时域分析**：基于传统信号处理的 SVM/HMM 模型
- **频域分析**：Wav2Vec2 + Transformer 混合模型
- **声音克隆**：Coqui XTTS-v2 零样本 TTS 模型

## 依赖项

### Node.js 依赖
```bash
npm install express multer
```

### Python 依赖
```bash
pip install torch torchaudio transformers TTS soundfile
```

## 环境变量

可选环境变量配置：

- `PYTHON_PATH`: Python 解释器路径（默认: `python`）
- `WAV2VEC_MODEL_NAME`: Wav2Vec2 模型名称
- `STT_SAMPLE_RATE`: 采样率（默认: 16000）
- `STT_MEL_BINS`: Mel 频谱 bins 数量（默认: 80）
- `STT_MEL_FRAMES`: Mel 频谱帧数（默认: 128）

## 文件结构

```
dsp-webui/
├── index.html          # 主页面
├── script.js           # 前端逻辑
├── style.css           # 样式表
├── server.js           # Node.js 服务器
├── stt_api.py          # 频域识别 API
├── time_api.py         # 时域识别 API
├── tts_api.py          # 语音合成 API
├── uploads/            # 临时上传目录
├── processed/          # 预处理文件目录
└── clone_output/       # 合成音频输出目录
```

## 注意事项

1. 首次使用声音克隆功能时，会自动下载 XTTS-v2 模型（约 2GB）
2. 参考音频建议长度为 3-10 秒，音质清晰
3. 服务器会在退出时自动清理临时文件
4. 建议使用 GPU 加速以获得更快的合成速度

## 故障排除

### 模型加载失败
- 确保已安装所有 Python 依赖
- 检查模型文件路径是否正确
- 确认有足够的磁盘空间

### 合成速度慢
- 检查是否有可用的 CUDA GPU
- 减少合成文本长度
- 考虑使用更短的参考音频

### 上传失败
- 确认文件格式为 WAV
- 检查文件大小限制
- 确保服务器有写入权限
