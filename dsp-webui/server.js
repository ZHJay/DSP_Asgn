const express = require('express');
const fs = require('fs');
const multer = require('multer');
const path = require('path');
const { spawn } = require('child_process');

const app = express();
const PORT = 3000;
const uploadDir = path.join(__dirname, 'uploads');
const processedDir = path.join(__dirname, 'processed');
const cloneOutputDir = path.join(__dirname, 'clone_output');

for (const dirPath of [uploadDir, processedDir, cloneOutputDir]) {
    if (!fs.existsSync(dirPath)) {
        fs.mkdirSync(dirPath, { recursive: true });
    }
}

// 配置文件上传
const storage = multer.diskStorage({
    destination: (req, file, cb) => {
        cb(null, uploadDir);
    },
    filename: (req, file, cb) => {
        cb(null, file.originalname);
    }
});
const upload = multer({ storage });

app.use(express.json());

// 静态文件服务器
app.use(express.static(path.join(__dirname)));

// 处理上传的 WAV 文件
app.post('/process', upload.single('wavFile'), async (req, res) => {
    if (!req.file) {
        return res.status(400).json({ message: '未收到任何文件' });
    }

    try {
        const sourcePath = req.file.path;
        const processedPath = path.join(processedDir, req.file.originalname);

        await preprocessWav(sourcePath, processedPath);

        console.log(`收到文件: ${req.file.originalname}`);

        res.json({
            message: '预处理已完成',
            fileName: req.file.originalname,
            processedFile: path.relative(__dirname, processedPath)
        });
    } catch (processingError) {
        console.error('处理文件时出错:', processingError);
        res.status(500).json({ message: '处理失败，请稍后重试。' });
    }
});

app.post('/time-domain', async (req, res) => {
    const { processedFile } = req.body || {};
    if (!processedFile) {
        return res.status(400).json({ message: '缺少 processedFile 参数' });
    }

    const safeFileName = path.basename(processedFile);
    const targetPath = path.join(processedDir, safeFileName);

    try {
        await fs.promises.access(targetPath, fs.constants.F_OK);
    } catch (accessError) {
        return res.status(404).json({ message: '目标文件不存在' });
    }

    try {
        const analysisResult = await 时域(targetPath);
        const digitConfidence = formatConfidence(analysisResult?.digit_confidence ?? analysisResult?.confidence);
        const summaryMessage = `识别成功: 数字 ${analysisResult?.predicted_digit ?? analysisResult?.digit ?? '未知'} (置信度: ${digitConfidence})`;

        res.json({
            message: summaryMessage,
            fileName: safeFileName,
            analysis: analysisResult
        });
    } catch (analysisError) {
        console.error('时域处理失败:', analysisError);
        res.status(500).json({ message: '时域处理失败，请稍后重试。' });
    }
});

app.post('/frequency-domain', async (req, res) => {
    const { processedFile } = req.body || {};
    if (!processedFile) {
        return res.status(400).json({ message: '缺少 processedFile 参数' });
    }

    const safeFileName = path.basename(processedFile);
    const targetPath = path.join(processedDir, safeFileName);

    try {
        await fs.promises.access(targetPath, fs.constants.F_OK);
    } catch (accessError) {
        return res.status(404).json({ message: '目标文件不存在' });
    }

    try {
        const analysisResult = await 频域(targetPath);
        const digitConfidence = formatConfidence(analysisResult?.digit_confidence);
        const speakerConfidence = formatConfidence(analysisResult?.speaker_confidence);
    const summaryMessage = `识别成功: 数字 ${analysisResult?.predicted_digit ?? '未知'} (置信度: ${digitConfidence})；说话人: ${analysisResult?.predicted_speaker ?? '未知'} (置信度: ${speakerConfidence})`;

        res.json({
            message: summaryMessage,
            fileName: safeFileName,
            analysis: analysisResult
        });
    } catch (analysisError) {
        console.error('频域处理失败:', analysisError);
        res.status(500).json({ message: '频域处理失败，请稍后重试。' });
    }
});

// 对话接口 - STT
app.post('/chat/stt', upload.single('audio'), async (req, res) => {
    if (!req.file) {
        return res.status(400).json({ message: '未收到音频文件' });
    }

    const { language = 'zh-cn' } = req.body;

    try {
        const audioPath = req.file.path;
        const result = await runChatSTT(audioPath, language);
        
        res.json({
            text: result.text,
            success: true
        });
    } catch (error) {
        console.error('STT失败:', error);
        res.status(500).json({ message: 'STT失败，请稍后重试。', error: error.message });
    }
});

// 对话接口 - LLM
app.post('/chat/llm', async (req, res) => {
    const { messages } = req.body;
    
    if (!messages || !Array.isArray(messages)) {
        return res.status(400).json({ message: '缺少消息历史' });
    }

    try {
        const result = await runChatLLM(messages);
        
        res.json({
            response: result.response,
            success: true
        });
    } catch (error) {
        console.error('LLM调用失败:', error);
        res.status(500).json({ message: 'LLM调用失败，请稍后重试。', error: error.message });
    }
});

// 对话接口 - TTS
app.post('/chat/tts', upload.single('referenceAudio'), async (req, res) => {
    if (!req.file) {
        return res.status(400).json({ message: '未收到参考音频文件' });
    }

    const { text, language = 'zh-cn' } = req.body;
    
    if (!text || !text.trim()) {
        return res.status(400).json({ message: '缺少要合成的文本' });
    }

    try {
        const referenceAudioPath = req.file.path;
        const timestamp = Date.now();
        const outputFileName = `chat_reply_${timestamp}.wav`;
        const outputPath = path.join(cloneOutputDir, outputFileName);

        await runChatTTS(text.trim(), referenceAudioPath, outputPath, language);

        const relativePath = path.relative(__dirname, outputPath);
        
        res.json({
            outputPath: relativePath,
            success: true
        });
    } catch (error) {
        console.error('TTS失败:', error);
        res.status(500).json({ message: 'TTS失败，请稍后重试。', error: error.message });
    }
});

// 声音克隆接口
app.post('/voice-clone', upload.single('referenceAudio'), async (req, res) => {
    if (!req.file) {
        return res.status(400).json({ message: '未收到参考音频文件' });
    }

    const { text, language = 'zh-cn' } = req.body;
    if (!text || !text.trim()) {
        return res.status(400).json({ message: '缺少要合成的文本' });
    }

    try {
        const referenceAudioPath = req.file.path;
        
        // 生成输出文件名：输入文件名（不含扩展名）+ 时间戳（精确到秒）
        const originalName = path.parse(req.file.originalname).name;
        const now = new Date();
        const timestamp = now.getFullYear() +
            String(now.getMonth() + 1).padStart(2, '0') +
            String(now.getDate()).padStart(2, '0') +
            '_' +
            String(now.getHours()).padStart(2, '0') +
            String(now.getMinutes()).padStart(2, '0') +
            String(now.getSeconds()).padStart(2, '0');
        const outputFileName = `${originalName}_${timestamp}.wav`;
        const outputPath = path.join(cloneOutputDir, outputFileName);

        await runVoiceClone(referenceAudioPath, text.trim(), outputPath, language);

        const relativePath = path.relative(__dirname, outputPath);
        console.log(`声音克隆完成: ${outputFileName}`);

        res.json({
            message: '声音克隆完成',
            outputPath: relativePath
        });
    } catch (error) {
        console.error('声音克隆失败:', error);
        res.status(500).json({ message: '声音克隆失败，请稍后重试。' });
    }
});

async function preprocessWav(inputPath, outputPath) {
    // TODO: 添加实际的预处理逻辑；目前仅复制文件
    await fs.promises.copyFile(inputPath, outputPath);
}

async function 时域(inputPath) {
    return runTimeDomainAnalysis(inputPath);
}

async function 频域(inputPath) {
    return runSttAnalysis(inputPath);
}

const server = app.listen(PORT, () => {
    console.log(`Server is running at http://localhost:${PORT}`);
});

let cleanedUp = false;

// 清理临时上传与处理目录，确保下次启动为干净状态
function cleanupTempDirs() {
    if (cleanedUp) {
        return;
    }

    cleanedUp = true;

    for (const dirPath of [uploadDir, processedDir, cloneOutputDir]) {
        try {
            if (!fs.existsSync(dirPath)) {
                continue;
            }

            fs.rmSync(dirPath, { recursive: true, force: true });
            fs.mkdirSync(dirPath, { recursive: true });
        } catch (cleanupError) {
            console.error('清理临时目录时出错:', cleanupError);
        }
    }
}

function handleShutdown(signal) {
    console.log(`收到 ${signal} 信号，正在退出服务器…`);

    server.close(() => {
        cleanupTempDirs();
        process.exit(0);
    });

    // 防止 server.close 卡住
    setTimeout(() => {
        cleanupTempDirs();
        process.exit(0);
    }, 5000).unref();
}

for (const signal of ['SIGINT', 'SIGTERM']) {
    process.once(signal, () => handleShutdown(signal));
}

process.on('exit', cleanupTempDirs);

function formatConfidence(value) {
    if (typeof value !== 'number' || Number.isNaN(value)) {
        return '未知';
    }

    return `${(value * 100).toFixed(2)}%`;
}

function runTimeDomainAnalysis(targetPath) {
    return executePythonScript('time_api.py', targetPath);
}

function runSttAnalysis(targetPath) {
    return executePythonScript('stt_api.py', targetPath);
}

function runChatSTT(audioPath, language) {
    return new Promise((resolve, reject) => {
        const pythonExecutable = process.env.PYTHON_PATH || 'python';
        const scriptPath = path.join(__dirname, 'chat_api.py');

        const child = spawn(pythonExecutable, [
            scriptPath,
            '--mode', 'stt',
            '--audio', audioPath,
            '--language', language
        ], {
            cwd: __dirname,
            windowsHide: true
        });

        let stdout = '';
        let stderr = '';

        child.stdout.on('data', (chunk) => {
            stdout += chunk.toString();
        });

        child.stderr.on('data', (chunk) => {
            stderr += chunk.toString();
        });

        child.on('error', (error) => {
            reject(error);
        });

        child.on('close', (code) => {
            if (code !== 0) {
                return reject(new Error(stderr || `STT 脚本以状态码 ${code} 退出`));
            }

            try {
                // 只解析最后一行的 JSON，忽略其他输出
                const lines = stdout.trim().split('\n');
                const jsonLine = lines[lines.length - 1];
                const parsed = JSON.parse(jsonLine);
                resolve(parsed);
            } catch (parseError) {
                reject(new Error(`解析 STT 输出失败: ${parseError.message}`));
            }
        });
    });
}

function runChatLLM(messages) {
    const messagesJson = JSON.stringify(messages);
    return new Promise((resolve, reject) => {
        const pythonExecutable = process.env.PYTHON_PATH || 'python';
        const scriptPath = path.join(__dirname, 'chat_api.py');

        const child = spawn(pythonExecutable, [
            scriptPath,
            '--mode', 'llm',
            '--messages', messagesJson
        ], {
            cwd: __dirname,
            windowsHide: true
        });

        let stdout = '';
        let stderr = '';

        child.stdout.on('data', (chunk) => {
            stdout += chunk.toString();
        });

        child.stderr.on('data', (chunk) => {
            stderr += chunk.toString();
        });

        child.on('error', (error) => {
            reject(error);
        });

        child.on('close', (code) => {
            if (code !== 0) {
                return reject(new Error(stderr || `LLM 脚本以状态码 ${code} 退出`));
            }

            try {
                // 只解析最后一行的 JSON，忽略其他输出
                const lines = stdout.trim().split('\n');
                const jsonLine = lines[lines.length - 1];
                const parsed = JSON.parse(jsonLine);
                resolve(parsed);
            } catch (parseError) {
                reject(new Error(`解析 LLM 输出失败: ${parseError.message}`));
            }
        });
    });
}

function runChatTTS(text, referenceAudioPath, outputPath, language) {
    return new Promise((resolve, reject) => {
        const pythonExecutable = process.env.PYTHON_PATH || 'python';
        const scriptPath = path.join(__dirname, 'chat_api.py');

        const child = spawn(pythonExecutable, [
            scriptPath,
            '--mode', 'tts',
            '--text', text,
            '--reference', referenceAudioPath,
            '--output', outputPath,
            '--language', language
        ], {
            cwd: __dirname,
            windowsHide: true
        });

        let stdout = '';
        let stderr = '';

        child.stdout.on('data', (chunk) => {
            stdout += chunk.toString();
        });

        child.stderr.on('data', (chunk) => {
            stderr += chunk.toString();
        });

        child.on('error', (error) => {
            reject(error);
        });

        child.on('close', (code) => {
            if (code !== 0) {
                return reject(new Error(stderr || `TTS 脚本以状态码 ${code} 退出`));
            }

            try {
                // 只解析最后一行的 JSON，忽略其他输出
                const lines = stdout.trim().split('\n');
                const jsonLine = lines[lines.length - 1];
                const parsed = JSON.parse(jsonLine);
                resolve(parsed);
            } catch (parseError) {
                reject(new Error(`解析 TTS 输出失败: ${parseError.message}`));
            }
        });
    });
}

function runVoiceClone(referenceAudioPath, text, outputPath, language) {
    return new Promise((resolve, reject) => {
        const pythonExecutable = process.env.PYTHON_PATH || 'python';
        const scriptPath = path.join(__dirname, 'tts_api.py');

        const child = spawn(pythonExecutable, [
            scriptPath,
            '--reference', referenceAudioPath,
            '--text', text,
            '--output', outputPath,
            '--language', language
        ], {
            cwd: __dirname,
            windowsHide: true
        });

        let stdout = '';
        let stderr = '';

        child.stdout.on('data', (chunk) => {
            stdout += chunk.toString();
        });

        child.stderr.on('data', (chunk) => {
            stderr += chunk.toString();
        });

        child.on('error', (error) => {
            reject(error);
        });

        child.on('close', (code) => {
            if (code !== 0) {
                return reject(new Error(stderr || `TTS 脚本以状态码 ${code} 退出`));
            }

            resolve({ success: true, stdout });
        });
    });
}

function executePythonScript(scriptName, targetPath) {
    return new Promise((resolve, reject) => {
        const pythonExecutable = process.env.PYTHON_PATH || 'python';
        const scriptPath = path.join(__dirname, scriptName);

        const child = spawn(pythonExecutable, [scriptPath, targetPath], {
            cwd: __dirname,
            windowsHide: true
        });

        let stdout = '';
        let stderr = '';

        child.stdout.on('data', (chunk) => {
            stdout += chunk.toString();
        });

        child.stderr.on('data', (chunk) => {
            stderr += chunk.toString();
        });

        child.on('error', (error) => {
            reject(error);
        });

        child.on('close', (code) => {
            if (code !== 0) {
                return reject(new Error(stderr || `Python 脚本以状态码 ${code} 退出`));
            }

            try {
                const trimmed = stdout.trim();
                if (!trimmed) {
                    return reject(new Error('Python 脚本未返回任何输出'));
                }

                const parsed = JSON.parse(trimmed);
                resolve(parsed);
            } catch (parseError) {
                reject(new Error(`解析 Python 输出失败: ${parseError.message}`));
            }
        });
    });
}