# DSP WebUI 项目工作流汇报

本文档详细介绍了 DSP 大作业 WebUI 部分的各个功能模块及其工作流。该系统基于 B/S 架构，前端使用 Vue.js，后端使用 Node.js (Express) 作为网关，底层核心算法由 Python 脚本及常驻服务提供支持。

## 1. 系统架构概览

系统主要由以下三部分组成：

1.  **前端 (Frontend)**:
    *   基于 HTML5 + Vue.js 3 (CDN) 构建的单页应用 (SPA)。
    *   提供文件上传、录音、实时交互和可视化展示功能。
    *   样式文件: `style.css`，逻辑文件: `script.js`。

2.  **后端网关 (Backend Gateway)**:
    *   基于 Node.js (Express) 的服务器 (`server.js`)，运行在端口 `3000`。
    *   负责静态资源托管、文件上传处理、以及调度 Python 脚本或转发请求给 Python 服务。

3.  **算法服务 (Algorithm Services)**:
    *   **脚本调用模式**: 针对一次性任务（如数字识别、智能锁验证），Node.js 通过 `child_process.spawn` 调用 Python 脚本 (`stt_api.py`, `lock_api.py`, `time_api.py`)。
    *   **常驻服务模式**: 针对高频且模型加载耗时的任务（如对话、克隆），使用 Flask 服务 (`chat_service.py`) 运行在端口 `5001`，常驻内存以提供低延迟响应。
    *   **外部 LLM 服务**: 对话功能依赖本地运行的大语言模型服务 (如 LM Studio)，默认端口 `1234`。

---

## 2. 功能模块工作流详解

### 2.1 数字识别 (Digital Recognition)

该功能用于对上传的音频文件进行时域和频域分析，识别其中的数字内容和说话人身份。

**工作流程:**

1.  **文件上传**:
    *   用户在前端选择 WAV 文件并点击“上传”。
    *   前端调用 `POST /process` 接口。
    *   后端将文件保存至 `uploads/` 目录，并复制到 `processed/` 目录作为预处理结果。

2.  **时域分析**:
    *   用户点击“时域分析”。
    *   前端调用 `POST /time-domain`。
    *   后端执行 `python time_api.py <filepath>`。
    *   Python 脚本计算音频的时域特征（如时长、采样率等），并返回 JSON 结果。

3.  **频域分析 (核心识别)**:
    *   用户点击“频域分析”。
    *   前端调用 `POST /frequency-domain`。
    *   后端执行 `python stt_api.py <filepath>`。
    *   **算法逻辑 (`stt_api.py`)**:
        *   加载预训练的 `SpeechTransformer` 模型 (基于 Wav2Vec2)。
        *   加载 `metrics.json` 获取数字和说话人的标签映射。
        *   对音频进行 Mel 频谱变换。
        *   模型推理，输出数字概率分布 (`digit_probabilities`) 和说话人概率分布 (`speaker_probabilities`)。
    *   前端接收结果并绘制置信度柱状图。

### 2.2 智能门锁 (Smart Lock)

该功能模拟声纹锁，验证说话人身份和口令密码是否匹配。

**工作流程:**

1.  **设置与输入**:
    *   用户在前端选择“主人” (Owner) 并设置“数字密码” (Passcode)。
    *   用户上传包含语音口令的音频文件。
    *   前端调用 `POST /smart-lock/verify`，附带音频文件及验证参数。

2.  **验证处理**:
    *   后端执行 `python lock_api.py --audio <path> --owner <owner> --passcode <code> --digits <len>`。
    *   **算法逻辑 (`lock_api.py`)**:
        *   **说话人识别**: 对整段音频进行特征提取，通过模型判断说话人身份，计算 Top-3 概率。
        *   **数字分割**: 使用能量阈值法 (`energy_segments`) 将音频切割成独立的数字片段。
        *   **数字识别**: 对每个切片分别进行数字识别。
        *   **逻辑判断**:
            *   `speaker_match`: 识别出的说话人 == 预设主人。
            *   `passcode_match`: 识别出的数字序列 == 预设密码。
            *   只有两者同时满足，才返回 `unlock: true`。

3.  **结果反馈**:
    *   前端根据返回的 JSON 显示“开锁成功”或“失败原因”，并展示详细的识别过程（每位数字的置信度）。

### 2.3 声音克隆 (Voice Cloning)

该功能允许用户上传或录制一段参考音频，输入文本，生成模仿该参考音色的语音。

**工作流程:**

1.  **提交请求**:
    *   用户上传/录制参考音频，输入文本，选择语言。
    *   前端调用 `POST /voice-clone`。

2.  **任务转发**:
    *   Node.js 后端接收请求，保存参考音频。
    *   后端调用内部函数 `runVoiceClone`，该函数实际上复用了 Chat Service 的 TTS 能力。
    *   请求被转发至 `http://localhost:5001/tts`。

3.  **语音合成**:
    *   `chat_service.py` 接收请求。
    *   使用预加载的 `XTTS` (Coqui TTS) 模型。
    *   以参考音频为 Condition，将文本合成为目标语音。
    *   生成的音频保存在 `clone_output/` 目录。

4.  **结果展示**:
    *   前端收到音频路径，提供在线播放和下载。

### 2.4 数字人对话 (Digital Human Chat)

该功能实现了一个全语音交互的 AI 助手，包含 语音识别(STT) -> 大模型思考(LLM) -> 语音合成(TTS) 的完整链路。

**工作流程:**

1.  **初始化**:
    *   用户上传参考音色（决定 AI 的声音）。
    *   确保 `chat_service.py` (端口 5001) 和 本地 LLM (端口 1234) 正在运行。

2.  **语音识别 (STT)**:
    *   用户录音或上传音频。
    *   前端调用 `POST /chat/stt` -> Node.js 转发 -> `chat_service.py` (`/stt`)。
    *   **算法**: 使用 `OpenAI Whisper` 模型将语音转写为文本。

3.  **大模型思考 (LLM)**:
    *   前端将识别出的文本（或用户输入的文本）加入对话历史。
    *   前端调用 `POST /chat/llm` -> Node.js 转发 -> `chat_service.py` (`/llm`)。
    *   **交互**: `chat_service.py` 将请求转发给本地 LLM 服务 (如 LM Studio 提供的 OpenAI 兼容接口)。
    *   **处理**: LLM 生成回复文本（可能包含 `<|im_start|>` 标签的思考过程）。

4.  **语音合成 (TTS)**:
    *   前端收到 LLM 回复后，提取主要内容（去除思考过程）。
    *   前端调用 `POST /chat/tts` -> Node.js 转发 -> `chat_service.py` (`/tts`)。
    *   **算法**: 使用 `XTTS` 模型，根据用户最初上传的参考音色，将 AI 的回复文本合成为语音。

5.  **交互完成**:
    *   前端播放合成的 AI 语音，并显示文字回复和思考过程。

### 2.5 前端 Vue.js 实现细节 (Frontend Implementation)

前端采用 **Vue.js 3 (Options API)** 开发，通过 CDN 引入，无需构建步骤，保持了项目的轻量化和易部署性。

#### 1. 核心设计模式
*   **单页应用 (SPA)**: 整个应用挂载在 `#app` 节点，通过 `activeTab` 状态控制不同功能模块 (`<main>` 标签) 的显示与隐藏 (`v-show`)，实现无刷新切换。
*   **响应式数据驱动**: 所有界面状态（文件选择、处理进度、图表数据、对话历史）均由 Vue 的 `data` 对象统一管理，UI 会随数据变化自动更新。

#### 2. 关键技术点
*   **音频录制与转换**:
    *   利用浏览器原生 `MediaRecorder` API 捕获麦克风输入 (WebM 格式)。
    *   使用 `AudioContext` 将 WebM 解码并重新编码为 WAV 格式 (16kHz, 单声道)，以满足后端 Python 算法的输入要求。
    *   相关代码位于 `startRecording` 和 `convertToWav` 方法中。
*   **动态可视化图表**:
    *   数字识别和说话人识别的置信度柱状图并非使用第三方库，而是通过 Vue 的列表渲染 (`v-for`) 和样式绑定 (`:style="{ height: ... }"`) 原生实现。
    *   这种方式轻量且易于定制，能够实时响应后端返回的概率数据。
*   **流式对话交互**:
    *   **状态机管理**: 对话过程被拆分为 `stt` (识别) -> `llm` (思考) -> `tts` (合成) 三个阶段，前端通过 `chatProcessingStep` 状态实时展示当前处理环节。
    *   **思维链展示**: 针对 DeepSeek 等推理模型输出的 `<|im_start|>` 标签，前端实现了专门的解析逻辑 (`extractThinkContent`)，支持折叠/展开 AI 的思考过程，提升用户体验。
    *   **自动滚动**: 使用 `this.$nextTick` 配合 `scrollTop` 属性，确保对话窗口始终滚动到最新消息。

#### 3. 状态管理结构
`script.js` 中的 `data` 对象主要包含以下几类状态：
*   **全局状态**: `activeTab` (当前标签页), `messages` (全局日志)。
*   **识别模块**: `selectedFile` (当前文件), `chartState` (图表数据), `timeFeatureGroups` (时域特征)。
*   **克隆模块**: `cloneReferenceFile` (参考音频), `cloneText` (文本), `isCloning` (合成状态)。
*   **对话模块**: `chatHistory` (消息列表), `chatProcessingStep` (流水线状态), `isRecording` (录音状态)。
*   **门锁模块**: `lockOwner` (当前主人), `lockResult` (验证结果)。

---

## 3. 依赖服务说明

为了使 WebUI 正常工作，需要启动以下服务：

| 服务名称 | 启动命令 | 端口 | 作用 |
| :--- | :--- | :--- | :--- |
| **WebUI Server** | `node server.js` | 3000 | 主入口，静态页面，API 网关 |
| **Chat Service** | `python chat_service.py` | 5001 | 提供 STT, TTS, LLM 代理功能的常驻服务 |
| **Local LLM** | (使用 LM Studio 等工具启动) | 1234 | 提供智能对话生成的推理服务 |

## 4. 目录结构关键文件

*   `dsp-webui/`
    *   `index.html`: 用户界面结构。
    *   `script.js`: 前端交互逻辑，API 调用。
    *   `server.js`: 后端服务器，路由分发。
    *   `chat_service.py`: 核心算法常驻服务 (Whisper + XTTS)。
    *   `stt_api.py`: 频域分析与数字识别脚本。
    *   `lock_api.py`: 智能门锁验证脚本。
    *   `time_api.py`: 时域分析脚本。
