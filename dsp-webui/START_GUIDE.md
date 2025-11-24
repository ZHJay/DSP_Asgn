# 服务启动指南

## 架构说明

为了解决模型重复加载的问题，现在采用了**双进程架构**：

1. **Chat Service (Python)** - 端口 5001
   - 常驻服务，启动时预加载所有模型
   - 提供 STT、LLM、TTS 的 HTTP API
   - 模型常驻内存，避免重复加载

2. **Web Server (Node.js)** - 端口 3000
   - 提供 Web UI 和文件上传
   - 转发请求到 Chat Service
   - 处理其他功能（数字识别、智能锁等）

## 启动步骤

### 1. 安装依赖

#### Python 依赖
```bash
cd /Users/zhanghjay/Desktop/DSP_Asgn/dsp-webui
pip install flask
```

#### Node.js 依赖
```bash
cd /Users/zhanghjay/Desktop/DSP_Asgn/dsp-webui
npm install
```

### 2. 启动服务（按顺序）

#### 终端 1: 启动 Chat Service
```bash
cd /Users/zhanghjay/Desktop/DSP_Asgn/dsp-webui
python chat_service.py
```

**预期输出：**
```
============================================================
启动 Chat Service (常驻服务模式)
端口: 5001
============================================================

============================================================
开始加载模型...
============================================================

[1/2] 加载 Whisper STT 模型 (small)...
✅ Whisper 模型加载完成 (X.XX秒)

[2/2] 加载 XTTS TTS 模型...
============================================================
Initialising Coqui XTTS-v2
============================================================
✅ XTTS 模型加载完成 (X.XX秒)

============================================================
✅ 所有模型加载完成！总耗时: X.XX秒
============================================================

🚀 Chat Service 已启动，监听端口 5001
============================================================

 * Running on all addresses (0.0.0.0)
 * Running on http://127.0.0.1:5001
```

**保持此终端运行，不要关闭！**

#### 终端 2: 启动 Web Server
```bash
cd /Users/zhanghjay/Desktop/DSP_Asgn/dsp-webui
node server.js
```

**预期输出：**
```
Server is running at http://localhost:3000

正在检查 Chat Service 状态...
提示: 请确保已在另一个终端运行: python chat_service.py

✅ Chat service 已就绪，模型已加载
✅ 服务已完全就绪！
```

### 3. 访问 Web UI

打开浏览器访问: http://localhost:3000

## 性能对比

### 改进前（每次spawn新进程）
- 首次 TTS 请求：**16-48秒**（CPU）或 **5-16秒**（GPU）
- 后续 TTS 请求：**11-33秒**（CPU）或 **3-11秒**（GPU）
- 每次请求都要重新加载模型

### 改进后（常驻服务）
- 首次 TTS 请求：**2-8秒**（仅推理时间）
- 后续 TTS 请求：**2-8秒**（稳定）
- 模型只加载一次，常驻内存

**性能提升：5-10倍！**

## 故障排查

### 问题1: Chat Service 启动失败
```
ImportError: No module named flask
```
**解决：** `pip install flask`

### 问题2: Web Server 报错 "Chat Service 未启动"
**原因：** Chat Service 未启动或正在加载模型
**解决：** 
1. 确保终端1已启动 chat_service.py
2. 等待模型加载完成（看到 "🚀 Chat Service 已启动"）
3. 重启 Web Server

### 问题3: TTS 仍然很慢
**可能原因：**
1. 使用 CPU 推理（查看 chat_service.py 输出的设备信息）
2. 文本太长（尝试缩短文本）
3. 系统资源不足

**优化建议：**
- 使用 GPU（如果有）
- 关闭其他占用资源的程序
- 使用更小的 Whisper 模型：`WHISPER_MODEL=tiny python chat_service.py`

## 环境变量配置（可选）

### Chat Service
```bash
# Whisper 模型大小: tiny, base, small, medium, large
export WHISPER_MODEL=small

# 本地 LLM 地址
export LOCAL_LLM_URL=http://localhost:1234/v1/chat/completions

# Chat Service 端口
export CHAT_SERVICE_PORT=5001
```

### Web Server
```bash
# Python 可执行文件路径
export PYTHON_PATH=python3
```

## 停止服务

1. 在终端2按 `Ctrl+C` 停止 Web Server
2. 在终端1按 `Ctrl+C` 停止 Chat Service

## 开发模式

如果需要修改 chat_service.py 并重启：
1. 在终端1按 `Ctrl+C`
2. 重新运行 `python chat_service.py`
3. 无需重启 Web Server
