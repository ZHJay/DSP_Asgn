const { createApp } = Vue;

createApp({
    data() {
        return {
            activeTab: 'recognition',
            selectedFile: null,
            lastProcessedFile: null,
            messages: [],
            isUploading: false,
            isProcessing: false,
            timeFeatureGroups: [],
            timeFeatureMeta: null,
            timeFeatureExpanded: false,
            cloneReferenceFile: null,
            cloneText: '',
            cloneLanguage: 'zh-cn',
            isCloning: false,
            cloneMessages: [],
            cloneOutputPath: null,
            cloneLogExpanded: false,
            recognitionLogExpanded: false,
            // Chat tab
            chatReferenceFile: null,
            chatLanguage: 'zh-cn',
            chatSystemPrompt: 'You are a friendly conversational assistant.',
            chatHistory: [],
            chatMessages: [],
            chatAudioFile: null,
            chatTextInput: '',
            isChatting: false,
            // 处理步骤状态
            chatProcessingStep: null, // 'stt', 'llm', 'tts', 'done'
            chatProcessingMessage: '',
            // 录音相关
            isRecording: false,
            hasRecording: false,
            chatRecordedBlob: null,
            mediaRecorder: null,
            audioChunks: [],
            recordingError: null,
            // Recognition tab recording
            isRecognitionRecording: false,
            hasRecognitionRecording: false,
            recognitionRecordedBlob: null,
            recognitionMediaRecorder: null,
            recognitionAudioChunks: [],
            recognitionRecordingError: null,
            // Clone tab recording
            isCloneRecording: false,
            hasCloneRecording: false,
            cloneRecordedBlob: null,
            cloneMediaRecorder: null,
            cloneAudioChunks: [],
            cloneRecordingError: null,
            // Smart Lock tab
            availableOwners: [],
            lockOwner: '',
            lockPasscode: '',
            lockDigits: 4,
            lockAudioFile: null,
            isLockRecording: false,
            hasLockRecording: false,
            lockRecordedBlob: null,
            lockMediaRecorder: null,
            lockAudioChunks: [],
            lockRecordingError: null,
            isVerifying: false,
            lockResult: null,
            lockMessages: [],
            lockLogExpanded: true,
            lockDetailsExpanded: false,
            chartState: {
                time: {
                    key: 'time',
                    title: '时域分析',
                    supportsSpeaker: false,
                    status: '等待分析',
                    digitBars: [],
                    speakerBars: []
                },
                frequency: {
                    key: 'frequency',
                    title: '频域分析',
                    supportsSpeaker: true,
                    status: '等待分析',
                    digitBars: [],
                    speakerBars: []
                }
            }
        };
    },
    computed: {
        canProcessTime() {
            return !!this.lastProcessedFile && !this.isUploading && !this.isProcessing;
        },
        canProcessFrequency() {
            return !!this.lastProcessedFile && !this.isUploading && !this.isProcessing;
        },
        chartSections() {
            return ['time', 'frequency']
                .map((key) => this.chartState[key])
                .filter(Boolean);
        }
    },
    created() {
        this.resetAllConfidenceTemplates();
        this.loadAvailableOwners();
    },
    watch: {
        activeTab(newTab) {
            if (newTab === 'lock' && this.availableOwners.length === 0) {
                this.loadAvailableOwners();
            }
        }
    },
    methods: {
        onFileChange(event) {
            const [file] = event.target.files || [];
            this.selectedFile = file || null;

            if (file) {
                this.lastProcessedFile = null;
                this.resetAllConfidenceTemplates();
                // 清除录音
                this.recognitionRecordedBlob = null;
                this.hasRecognitionRecording = false;
            }
        },
        appendMessage(message) {
            this.messages.push(message);
        },
        async parseResponse(response) {
            const rawPayload = await response.text();
            let parsedPayload;

            if (rawPayload) {
                try {
                    parsedPayload = JSON.parse(rawPayload);
                } catch (error) {
                    console.warn('响应不是 JSON，已回退为纯文本。', error);
                }
            }

            return { rawPayload, parsedPayload };
        },
        resetFileInput() {
            const input = this.$refs.fileInput;
            if (input) {
                input.value = '';
            }

            this.selectedFile = null;
        },
        async handleUpload() {
            const fileToUpload = this.selectedFile || this.recognitionRecordedBlob;
            if (!fileToUpload) {
                alert('请上传或录制一个 WAV 文件！');
                return;
            }

            this.appendMessage('正在上传并预处理，请稍候...');
            this.isUploading = true;

            const formData = new FormData();
            formData.append('wavFile', fileToUpload, 'recording.wav');

            try {
                const response = await fetch('/process', {
                    method: 'POST',
                    body: formData
                });

                const { rawPayload, parsedPayload } = await this.parseResponse(response);

                if (response.ok) {
                    const successMessage = parsedPayload?.message || rawPayload || '处理成功，但未收到详情。';
                    const fileNameHint = parsedPayload?.fileName ? `（文件: ${parsedPayload.fileName}）` : '';
                    const processedHint = parsedPayload?.processedFile ? `（已保存至: ${parsedPayload.processedFile}）` : '';
                    this.appendMessage(`处理结果: ${successMessage}${fileNameHint}${processedHint}`);

                    if (parsedPayload?.processedFile) {
                        this.lastProcessedFile = parsedPayload.processedFile;
                    } else {
                        this.lastProcessedFile = null;
                    }

                    this.resetAllConfidenceTemplates();
                } else {
                    const failureMessage = parsedPayload?.message || rawPayload || '处理失败，请重试！';
                    this.appendMessage(`处理失败: ${failureMessage}`);
                    this.lastProcessedFile = null;
                    this.resetAllConfidenceTemplates();
                }
            } catch (error) {
                console.error('上传时发生错误', error);
                this.appendMessage(`发生错误，请检查后台服务！(${error.message})`);
                this.lastProcessedFile = null;
                this.resetAllConfidenceTemplates();
            } finally {
                this.isUploading = false;
                this.resetFileInput();
                // 清除录音
                this.recognitionRecordedBlob = null;
                this.hasRecognitionRecording = false;
            }
        },
        async requestDomain(endpoint, label, domainKey) {
            if (!this.lastProcessedFile) {
                this.appendMessage(`请先完成预处理后再进行${label}。`);
                return;
            }

            this.appendMessage(`正在进行${label}，请稍候...`);
            this.isProcessing = true;
            this.setChartStatus(domainKey, '处理中…');

            try {
                const response = await fetch(endpoint, {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({ processedFile: this.lastProcessedFile })
                });

                const { rawPayload, parsedPayload } = await this.parseResponse(response);

                if (response.ok) {
                    const message = parsedPayload?.message || rawPayload || `${label}完成，但未收到详情。`;
                    const fileNameHint = parsedPayload?.fileName ? `（文件: ${parsedPayload.fileName}）` : '';
                    this.appendMessage(`${label}结果: ${message}${fileNameHint}`);

                    const analysis = parsedPayload?.analysis;
                    if (analysis) {
                        const digitSummary = (analysis.digit_probabilities || []).slice(0, 3);
                        if (digitSummary.length > 0) {
                            const digitText = digitSummary.map(({ digit, probability }) => {
                                const percent = typeof probability === 'number' ? `${(probability * 100).toFixed(2)}%` : '未知';
                                return `${digit}: ${percent}`;
                            }).join('，');
                            this.appendMessage(`数字概率Top3: ${digitText}`);
                        }

                        const speakerSummary = (analysis.speaker_probabilities || []).slice(0, 3);
                        if (speakerSummary.length > 0) {
                            const speakerText = speakerSummary.map(({ speaker, probability }) => {
                                const percent = typeof probability === 'number' ? `${(probability * 100).toFixed(2)}%` : '未知';
                                return `${speaker}: ${percent}`;
                            }).join('，');
                            this.appendMessage(`说话人概率Top3: ${speakerText}`);
                        }

                        this.updateConfidenceCharts(domainKey, analysis);
                    } else {
                        this.resetDomainTemplate(domainKey, '未返回数据');
                        if (domainKey === 'time') {
                            this.resetTimeFeatureGroups();
                        }
                    }
                } else {
                    const failureMessage = parsedPayload?.message || rawPayload || `${label}失败，请重试！`;
                    this.appendMessage(`${label}失败: ${failureMessage}`);
                    this.setChartStatus(domainKey, '分析失败');
                    if (domainKey === 'time') {
                        this.resetTimeFeatureGroups();
                    }
                }
            } catch (error) {
                console.error(`${label}时发生错误`, error);
                this.appendMessage(`${label}时发生错误，请检查后台服务！(${error.message})`);
                this.setChartStatus(domainKey, '分析失败');
                if (domainKey === 'time') {
                    this.resetTimeFeatureGroups();
                }
            } finally {
                this.isProcessing = false;
            }
        },
        processTimeDomain() {
            this.requestDomain('/time-domain', '时域分析', 'time');
        },
        processFrequencyDomain() {
            this.requestDomain('/frequency-domain', '频域分析', 'frequency');
        },
        getSection(domainKey) {
            return this.chartState?.[domainKey] ?? null;
        },
        setChartStatus(domainKey, status) {
            const section = this.getSection(domainKey);
            if (!section) {
                return;
            }

            section.status = status;
        },
        formatTimestamp() {
            return new Date().toLocaleTimeString([], {
                hour: '2-digit',
                minute: '2-digit',
                second: '2-digit'
            });
        },
        prepareBars(entries, { labelKey, fallbackLabel, limit = 5 }) {
            if (!Array.isArray(entries)) {
                return [];
            }

            return entries
                .map((entry, index) => {
                    const candidateProbabilities = [
                        entry?.probability,
                        entry?.confidence,
                        entry?.score,
                        entry?.value
                    ];
                    const rawProbability = candidateProbabilities.find((item) => typeof item === 'number' && !Number.isNaN(item));

                    if (typeof rawProbability !== 'number') {
                        return null;
                    }

                    const clampedProbability = Math.max(0, Math.min(1, rawProbability));
                    const rawLabel = entry?.[labelKey] ?? entry?.label;
                    const label = rawLabel != null && rawLabel !== '' ? rawLabel : `${fallbackLabel}${index + 1}`;

                    const percentValue = (clampedProbability * 100).toFixed(2);
                    const targetHeight = `${percentValue}%`;

                    return {
                        label: String(label),
                        probability: clampedProbability,
                        displayPercent: `${percentValue}%`,
                        barHeight: '0%',
                        targetHeight
                    };
                })
                .filter(Boolean)
                .sort((a, b) => b.probability - a.probability)
                .slice(0, limit);
        },
        setDigitProbabilities(domainKey, entries = []) {
            const section = this.getSection(domainKey);
            if (!section) {
                return;
            }

            const hasEntries = Array.isArray(entries) && entries.length > 0;
            section.digitBars = hasEntries
                ? this.prepareBars(entries, {
                    labelKey: 'digit',
                    fallbackLabel: '数字',
                    limit: 10
                })
                : this.createDigitTemplate();

            this.animateBars(domainKey, 'digitBars');
        },
        setSpeakerProbabilities(domainKey, entries = []) {
            const section = this.getSection(domainKey);
            if (!section || !section.supportsSpeaker) {
                return;
            }

            const hasEntries = Array.isArray(entries) && entries.length > 0;
            section.speakerBars = hasEntries
                ? this.prepareBars(entries, {
                    labelKey: 'speaker',
                    fallbackLabel: '说话人',
                    limit: 10
                })
                : this.createSpeakerTemplate();

            this.animateBars(domainKey, 'speakerBars');
        },
        updateConfidenceCharts(domainKey, analysis) {
            const section = this.getSection(domainKey);
            if (!section) {
                return;
            }

            if (!analysis) {
                this.resetDomainTemplate(domainKey, '未返回数据');
                if (domainKey === 'time') {
                    this.resetTimeFeatureGroups();
                }
                return;
            }

            const digitCandidates = Array.isArray(analysis.digit_probabilities) && analysis.digit_probabilities.length > 0
                ? analysis.digit_probabilities
                : this.buildFallbackEntries(analysis, {
                    probabilityKeys: ['digit_confidence', 'confidence'],
                    labelKeys: ['predicted_digit', 'digit'],
                    targetLabelKey: 'digit',
                    fallbackLabel: '结果'
                });

            this.setDigitProbabilities(domainKey, digitCandidates);

            if (section.supportsSpeaker) {
                const speakerCandidates = Array.isArray(analysis.speaker_probabilities) && analysis.speaker_probabilities.length > 0
                    ? analysis.speaker_probabilities
                    : this.buildFallbackEntries(analysis, {
                        probabilityKeys: ['speaker_confidence'],
                        labelKeys: ['predicted_speaker', 'speaker'],
                        targetLabelKey: 'speaker',
                        fallbackLabel: '说话人'
                    });

                this.setSpeakerProbabilities(domainKey, speakerCandidates);
            }

            this.setChartStatus(domainKey, `已更新 ${this.formatTimestamp()}`);

            if (domainKey === 'time') {
                this.updateTimeFeatureGroups(analysis.time_features);
            }
        },
        buildFallbackEntries(source, { probabilityKeys = [], labelKeys = [], targetLabelKey = 'label', fallbackLabel = '项' }) {
            if (!source) {
                return [];
            }

            const probabilityCandidate = probabilityKeys
                .map((key) => source?.[key])
                .find((value) => typeof value === 'number' && !Number.isNaN(value));

            if (typeof probabilityCandidate !== 'number') {
                return [];
            }

            const labelCandidate = labelKeys
                .map((key) => source?.[key])
                .find((value) => value != null && value !== '');

            return [
                {
                    probability: probabilityCandidate,
                    [targetLabelKey]: labelCandidate ?? fallbackLabel,
                    label: labelCandidate ?? fallbackLabel
                }
            ];
        },
        resetDomainTemplate(domainKey, status = '等待分析') {
            const section = this.getSection(domainKey);
            if (!section) {
                return;
            }

            section.digitBars = this.createDigitTemplate();
            section.speakerBars = section.supportsSpeaker ? this.createSpeakerTemplate() : [];
            this.setChartStatus(domainKey, status);
            this.animateBars(domainKey, 'digitBars');

            if (section.supportsSpeaker) {
                this.animateBars(domainKey, 'speakerBars');
            }
        },
        resetAllConfidenceTemplates() {
            // 初始化两种分析的占位条形图，保持可视化布局独立
            ['time', 'frequency'].forEach((domainKey) => {
                this.resetDomainTemplate(domainKey, '等待分析');
            });
            this.resetTimeFeatureGroups();
        },
        createDigitTemplate() {
            return Array.from({ length: 10 }, (_, index) => ({
                label: `数字${index}`,
                probability: 0,
                displayPercent: '--',
                barHeight: '0%',
                targetHeight: '0%'
            }));
        },
        createSpeakerTemplate(count = 5) {
            return Array.from({ length: count }, (_, index) => ({
                label: `说话人${index + 1}`,
                probability: 0,
                displayPercent: '--',
                barHeight: '0%',
                targetHeight: '0%'
            }));
        },
        animateBars(domainKey, collectionKey) {
            this.$nextTick(() => {
                const section = this.getSection(domainKey);
                if (!section) {
                    return;
                }

                const bars = section[collectionKey];
                requestAnimationFrame(() => {
                    if (!Array.isArray(bars)) {
                        return;
                    }

                    bars.forEach((bar) => {
                        bar.barHeight = bar.targetHeight || '0%';
                    });
                });
            });
        },
        resetTimeFeatureGroups() {
            this.timeFeatureGroups = [];
            this.timeFeatureMeta = null;
            this.timeFeatureExpanded = false;
        },
        updateTimeFeatureGroups(summary) {
            if (!summary) {
                this.resetTimeFeatureGroups();
                return;
            }

            const meta = summary.meta || {};
            this.timeFeatureMeta = {
                sampleRate: meta.sample_rate ?? null,
                durationSeconds: meta.duration_seconds ?? null,
                frameCount: meta.frame_count ?? null
            };

            const groups = Array.isArray(summary.groups) ? summary.groups : [];
            this.timeFeatureGroups = groups.map((group, groupIndex) => ({
                key: group.key || `group-${groupIndex}`,
                title: group.title || `特征组 ${groupIndex + 1}`,
                metrics: Array.isArray(group.metrics)
                    ? group.metrics.map((metric, metricIndex) => ({
                        label: metric.label || `指标 ${metricIndex + 1}`,
                        unit: metric.unit || '',
                        value: typeof metric.value === 'number' ? metric.value : null
                    }))
                    : []
            }));

            if (this.timeFeatureGroups.length > 0) {
                this.timeFeatureExpanded = false;
            }
        },
        formatFeatureMetric(value, unit = '') {
            if (value == null || Number.isNaN(value)) {
                return '--';
            }

            const magnitude = Math.abs(value);
            const decimals = magnitude >= 100 ? 0 : magnitude >= 10 ? 1 : 2;
            const formatted = value.toFixed(decimals);
            return unit ? `${formatted}${unit}` : formatted;
        },
        formatDuration(seconds) {
            if (typeof seconds !== 'number' || Number.isNaN(seconds)) {
                return '--';
            }

            if (seconds < 1) {
                return `${(seconds * 1000).toFixed(0)} ms`;
            }

            return `${seconds.toFixed(2)} s`;
        },
        toggleTimeFeatureSection() {
            this.timeFeatureExpanded = !this.timeFeatureExpanded;
        },
        onCloneReferenceChange(event) {
            const [file] = event.target.files || [];
            this.cloneReferenceFile = file || null;
            if (file) {
                // 清除录音
                this.cloneRecordedBlob = null;
                this.hasCloneRecording = false;
            }
        },
        appendCloneMessage(message) {
            this.cloneMessages.push(message);
        },
        async handleClone() {
            const referenceAudio = this.cloneReferenceFile || this.cloneRecordedBlob;
            if (!referenceAudio) {
                alert('请选择或录制参考音频文件！');
                return;
            }
            if (!this.cloneText.trim()) {
                alert('请输入要合成的文本！');
                return;
            }

            this.appendCloneMessage('正在上传并合成，请稍候...');
            this.isCloning = true;
            this.cloneOutputPath = null;

            const formData = new FormData();
            formData.append('referenceAudio', referenceAudio, 'reference.wav');
            formData.append('text', this.cloneText.trim());
            formData.append('language', this.cloneLanguage);

            try {
                const response = await fetch('/voice-clone', {
                    method: 'POST',
                    body: formData
                });

                const { rawPayload, parsedPayload } = await this.parseResponse(response);

                if (response.ok) {
                    const message = parsedPayload?.message || rawPayload || '合成成功';
                    this.appendCloneMessage(`✅ ${message}`);
                    
                    if (parsedPayload?.outputPath) {
                        this.cloneOutputPath = '/' + parsedPayload.outputPath;
                    }
                } else {
                    const errorMessage = parsedPayload?.message || rawPayload || '合成失败';
                    this.appendCloneMessage(`❌ ${errorMessage}`);
                }
            } catch (error) {
                console.error('声音克隆时发生错误', error);
                this.appendCloneMessage(`❌ 发生错误: ${error.message}`);
            } finally {
                this.isCloning = false;
            }
        },
        onChatReferenceChange(event) {
            const [file] = event.target.files || [];
            this.chatReferenceFile = file || null;
        },
        onChatAudioChange(event) {
            const [file] = event.target.files || [];
            this.chatAudioFile = file || null;
        },
        resetChat() {
            this.chatHistory = [];
            this.chatMessages = [{ role: 'system', content: this.chatSystemPrompt }];
            this.chatAudioFile = null;
            this.chatTextInput = '';
            this.chatRecordedBlob = null;
            this.hasRecording = false;
            this.recordingError = null;
            if (this.$refs.chatAudioInput) {
                this.$refs.chatAudioInput.value = '';
            }
        },
        async toggleRecording() {
            if (this.isRecording) {
                // 停止录音
                this.stopRecording();
            } else {
                // 开始录音
                await this.startRecording();
            }
        },
        async startRecording() {
            try {
                this.recordingError = null;
                
                // 请求麦克风权限
                const stream = await navigator.mediaDevices.getUserMedia({ 
                    audio: {
                        channelCount: 1,
                        sampleRate: 16000,
                        echoCancellation: true,
                        noiseSuppression: true
                    }
                });
                
                // 创建MediaRecorder
                const options = { mimeType: 'audio/webm' };
                this.mediaRecorder = new MediaRecorder(stream, options);
                this.audioChunks = [];
                
                this.mediaRecorder.ondataavailable = (event) => {
                    if (event.data.size > 0) {
                        this.audioChunks.push(event.data);
                    }
                };
                
                this.mediaRecorder.onstop = async () => {
                    // 合并音频数据
                    const audioBlob = new Blob(this.audioChunks, { type: 'audio/webm' });
                    
                    // 转换为WAV格式
                    try {
                        const wavBlob = await this.convertToWav(audioBlob);
                        this.chatRecordedBlob = wavBlob;
                        this.hasRecording = true;
                        this.chatAudioFile = null; // 清除文件上传
                        if (this.$refs.chatAudioInput) {
                            this.$refs.chatAudioInput.value = '';
                        }
                    } catch (error) {
                        console.error('音频转换失败:', error);
                        this.recordingError = '音频转换失败，请重试';
                        this.hasRecording = false;
                    }
                    
                    // 停止所有轨道
                    stream.getTracks().forEach(track => track.stop());
                };
                
                this.mediaRecorder.start();
                this.isRecording = true;
                
            } catch (error) {
                console.error('麦克风访问失败:', error);
                if (error.name === 'NotAllowedError') {
                    this.recordingError = '麦克风权限被拒绝，请在浏览器设置中允许访问麦克风';
                } else if (error.name === 'NotFoundError') {
                    this.recordingError = '未找到麦克风设备';
                } else {
                    this.recordingError = `麦克风访问失败: ${error.message}`;
                }
            }
        },
        stopRecording() {
            if (this.mediaRecorder && this.mediaRecorder.state !== 'inactive') {
                this.mediaRecorder.stop();
                this.isRecording = false;
            }
        },
        async convertToWav(webmBlob) {
            // 使用Web Audio API将WebM转换为WAV
            const arrayBuffer = await webmBlob.arrayBuffer();
            const audioContext = new (window.AudioContext || window.webkitAudioContext)();
            const audioBuffer = await audioContext.decodeAudioData(arrayBuffer);
            
            // 转换为WAV格式
            const wavBuffer = this.audioBufferToWav(audioBuffer);
            return new Blob([wavBuffer], { type: 'audio/wav' });
        },
        audioBufferToWav(audioBuffer) {
            const numChannels = audioBuffer.numberOfChannels;
            const sampleRate = audioBuffer.sampleRate;
            const length = audioBuffer.length * numChannels * 2;
            const buffer = new ArrayBuffer(44 + length);
            const view = new DataView(buffer);
            
            // WAV文件头
            const writeString = (offset, string) => {
                for (let i = 0; i < string.length; i++) {
                    view.setUint8(offset + i, string.charCodeAt(i));
                }
            };
            
            writeString(0, 'RIFF');
            view.setUint32(4, 36 + length, true);
            writeString(8, 'WAVE');
            writeString(12, 'fmt ');
            view.setUint32(16, 16, true);
            view.setUint16(20, 1, true);
            view.setUint16(22, numChannels, true);
            view.setUint32(24, sampleRate, true);
            view.setUint32(28, sampleRate * numChannels * 2, true);
            view.setUint16(32, numChannels * 2, true);
            view.setUint16(34, 16, true);
            writeString(36, 'data');
            view.setUint32(40, length, true);
            
            // 写入音频数据
            const offset = 44;
            const channels = [];
            for (let i = 0; i < numChannels; i++) {
                channels.push(audioBuffer.getChannelData(i));
            }
            
            let index = offset;
            for (let i = 0; i < audioBuffer.length; i++) {
                for (let channel = 0; channel < numChannels; channel++) {
                    const sample = Math.max(-1, Math.min(1, channels[channel][i]));
                    view.setInt16(index, sample < 0 ? sample * 0x8000 : sample * 0x7FFF, true);
                    index += 2;
                }
            }
            
            return buffer;
        },
        removeThinkTags(text) {
            // 移除<think>...</think>标签及其内容（支持多个think标签）
            // 使用全局匹配和非贪婪模式，支持换行符
            let cleaned = text.replace(/<think>[\s\S]*?<\/think>/gi, '');
            
            // 移除可能残留的单独标签
            cleaned = cleaned.replace(/<\/?think>/gi, '');
            
            // 清理多余的空白字符
            cleaned = cleaned.trim();
            
            // 如果清理后内容为空，返回默认提示
            return cleaned || '[AI正在思考中]';
        },
        async sendChatMessage() {
            if (!this.chatReferenceFile) {
                alert('请先上传参考音色！');
                return;
            }

            if (!this.chatAudioFile && !this.chatRecordedBlob && !this.chatTextInput.trim()) {
                alert('请上传音频、录制音频或输入文本！');
                return;
            }

            this.isChatting = true;
            this.chatProcessingStep = null;
            this.chatProcessingMessage = '';
            
            try {
                let userText = '';

                // 1. 如果有音频文件或录音，先进行STT
                const audioToTranscribe = this.chatAudioFile || this.chatRecordedBlob;
                if (audioToTranscribe) {
                    this.chatProcessingStep = 'stt';
                    this.chatProcessingMessage = '正在识别语音内容...';
                    
                    const sttFormData = new FormData();
                    sttFormData.append('audio', audioToTranscribe, 'recording.wav');
                    sttFormData.append('language', this.chatLanguage);

                    const sttResponse = await fetch('/chat/stt', {
                        method: 'POST',
                        body: sttFormData
                    });

                    if (!sttResponse.ok) {
                        const error = await sttResponse.json();
                        throw new Error(error.message || 'STT失败');
                    }

                    const sttResult = await sttResponse.json();
                    userText = sttResult.text || '[未识别]';
                    this.chatProcessingMessage = `✓ 识别完成: ${userText.substring(0, 20)}${userText.length > 20 ? '...' : ''}`;
                } else {
                    userText = this.chatTextInput.trim();
                }

                // 添加用户消息到界面
                this.chatHistory.push({
                    role: 'user',
                    content: userText,
                    time: this.formatTimestamp()
                });

                // 滚动到底部
                this.$nextTick(() => {
                    const chatHistoryEl = this.$refs.chatHistory;
                    if (chatHistoryEl) {
                        chatHistoryEl.scrollTop = chatHistoryEl.scrollHeight;
                    }
                });

                // 2. 调用LLM
                this.chatProcessingStep = 'llm';
                this.chatProcessingMessage = 'AI正在思考回复...';
                
                if (this.chatMessages.length === 0) {
                    this.chatMessages = [{ role: 'system', content: this.chatSystemPrompt }];
                }
                this.chatMessages.push({ role: 'user', content: userText });

                const llmResponse = await fetch('/chat/llm', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ messages: this.chatMessages })
                });

                if (!llmResponse.ok) {
                    const error = await llmResponse.json();
                    throw new Error(error.message || 'LLM调用失败');
                }

                const llmResult = await llmResponse.json();
                const assistantText = llmResult.response || '[无回复]';
                
                this.chatMessages.push({ role: 'assistant', content: assistantText });
                this.chatProcessingMessage = `✓ AI回复完成: ${assistantText.substring(0, 30).replace(/<think>.*?<\/think>/g, '')}${assistantText.length > 30 ? '...' : ''}`;

                // 过滤掉<think>标签及其内容，只保留实际回复用于TTS
                const textForTTS = this.removeThinkTags(assistantText);

                // 3. 使用TTS合成语音
                this.chatProcessingStep = 'tts';
                this.chatProcessingMessage = '正在合成语音...';
                
                const ttsFormData = new FormData();
                ttsFormData.append('referenceAudio', this.chatReferenceFile);
                ttsFormData.append('text', textForTTS);
                ttsFormData.append('language', this.chatLanguage);

                const ttsResponse = await fetch('/chat/tts', {
                    method: 'POST',
                    body: ttsFormData
                });

                let audioPath = null;
                if (ttsResponse.ok) {
                    const ttsResult = await ttsResponse.json();
                    audioPath = '/' + ttsResult.outputPath;
                    this.chatProcessingMessage = '✓ 语音合成完成';
                }

                // 添加AI回复到界面
                this.chatHistory.push({
                    role: 'assistant',
                    content: assistantText,
                    audioPath: audioPath,
                    time: this.formatTimestamp()
                });
                
                // 标记完成
                this.chatProcessingStep = 'done';
                this.chatProcessingMessage = '✓ 对话完成';

                // 清空输入
                this.chatAudioFile = null;
                this.chatRecordedBlob = null;
                this.hasRecording = false;
                this.chatTextInput = '';
                if (this.$refs.chatAudioInput) {
                    this.$refs.chatAudioInput.value = '';
                }

                // 再次滚动到底部
                this.$nextTick(() => {
                    const chatHistoryEl = this.$refs.chatHistory;
                    if (chatHistoryEl) {
                        chatHistoryEl.scrollTop = chatHistoryEl.scrollHeight;
                    }
                });

            } catch (error) {
                console.error('对话处理失败:', error);
                this.chatProcessingStep = null;
                this.chatProcessingMessage = '';
                alert(`对话失败: ${error.message}`);
            } finally {
                this.isChatting = false;
                // 延迟清除处理状态，让用户看到完成提示
                setTimeout(() => {
                    this.chatProcessingStep = null;
                    this.chatProcessingMessage = '';
                }, 2000);
            }
        },
        // ====== Smart Lock Methods ======
        async loadAvailableOwners() {
            try {
                const response = await fetch('/smart-lock/owners');
                if (response.ok) {
                    const result = await response.json();
                    this.availableOwners = result.owners || [];
                }
            } catch (error) {
                console.error('加载主人列表失败:', error);
                this.availableOwners = ['bck', 'cqc', 'xsq', 'zhj']; // 备用列表
            }
        },
        onLockAudioChange(event) {
            const [file] = event.target.files || [];
            this.lockAudioFile = file || null;
            if (file) {
                this.lockRecordedBlob = null;
                this.hasLockRecording = false;
            }
        },
        async toggleLockRecording() {
            if (this.isLockRecording) {
                this.stopLockRecording();
            } else {
                await this.startLockRecording();
            }
        },
        async startLockRecording() {
            try {
                this.lockRecordingError = null;
                
                const stream = await navigator.mediaDevices.getUserMedia({ 
                    audio: {
                        channelCount: 1,
                        sampleRate: 16000,
                        echoCancellation: true,
                        noiseSuppression: true
                    }
                });
                
                const options = { mimeType: 'audio/webm' };
                this.lockMediaRecorder = new MediaRecorder(stream, options);
                this.lockAudioChunks = [];
                
                this.lockMediaRecorder.ondataavailable = (event) => {
                    if (event.data.size > 0) {
                        this.lockAudioChunks.push(event.data);
                    }
                };
                
                this.lockMediaRecorder.onstop = async () => {
                    const audioBlob = new Blob(this.lockAudioChunks, { type: 'audio/webm' });
                    
                    try {
                        const wavBlob = await this.convertToWav(audioBlob);
                        this.lockRecordedBlob = wavBlob;
                        this.hasLockRecording = true;
                        this.lockAudioFile = null;
                        if (this.$refs.lockAudioInput) {
                            this.$refs.lockAudioInput.value = '';
                        }
                    } catch (error) {
                        console.error('音频转换失败:', error);
                        this.lockRecordingError = '音频转换失败，请重试';
                        this.hasLockRecording = false;
                    }
                    
                    stream.getTracks().forEach(track => track.stop());
                };
                
                this.lockMediaRecorder.start();
                this.isLockRecording = true;
                
            } catch (error) {
                console.error('麦克风访问失败:', error);
                if (error.name === 'NotAllowedError') {
                    this.lockRecordingError = '麦克风权限被拒绝';
                } else if (error.name === 'NotFoundError') {
                    this.lockRecordingError = '未找到麦克风设备';
                } else {
                    this.lockRecordingError = `麦克风访问失败: ${error.message}`;
                }
            }
        },
        stopLockRecording() {
            if (this.lockMediaRecorder && this.lockMediaRecorder.state !== 'inactive') {
                this.lockMediaRecorder.stop();
                this.isLockRecording = false;
            }
        },
        appendLockMessage(message) {
            this.lockMessages.push(message);
        },
        async verifyLock() {
            if (!this.lockOwner || !this.lockPasscode) {
                alert('请设置主人和密码！');
                return;
            }
            
            const audioToVerify = this.lockAudioFile || this.lockRecordedBlob;
            if (!audioToVerify) {
                alert('请上传音频或录制音频！');
                return;
            }
            
            this.isVerifying = true;
            this.lockResult = null;
            this.appendLockMessage(`开始验证: 主人=${this.lockOwner}, 密码=${this.lockPasscode}`);
            
            try {
                const formData = new FormData();
                formData.append('audio', audioToVerify, 'lock_audio.wav');
                formData.append('owner', this.lockOwner);
                formData.append('passcode', this.lockPasscode);
                formData.append('digits', this.lockDigits.toString());
                
                const response = await fetch('/smart-lock/verify', {
                    method: 'POST',
                    body: formData
                });
                
                const result = await response.json();
                
                if (response.ok) {
                    this.lockResult = result;
                    
                    if (result.unlock) {
                        this.appendLockMessage(`✅ 验证成功！识别: ${result.recognized_speaker} / ${result.recognized_digits}`);
                    } else {
                        let reason = '';
                        if (!result.speaker_match && !result.passcode_match) {
                            reason = '身份和密码均不匹配';
                        } else if (!result.speaker_match) {
                            reason = '身份不匹配';
                        } else {
                            reason = '密码不匹配';
                        }
                        this.appendLockMessage(`❌ 验证失败：${reason}`);
                    }
                } else {
                    this.appendLockMessage(`❌ 验证出错: ${result.message || '未知错误'}`);
                }
                
            } catch (error) {
                console.error('智能锁验证失败:', error);
                this.appendLockMessage(`❌ 验证失败: ${error.message}`);
            } finally {
                this.isVerifying = false;
            }
        },
        // ====== Recognition Recording Methods ======
        async toggleRecognitionRecording() {
            if (this.isRecognitionRecording) {
                this.stopRecognitionRecording();
            } else {
                await this.startRecognitionRecording();
            }
        },
        async startRecognitionRecording() {
            try {
                this.recognitionRecordingError = null;
                
                const stream = await navigator.mediaDevices.getUserMedia({ 
                    audio: {
                        channelCount: 1,
                        sampleRate: 16000,
                        echoCancellation: true,
                        noiseSuppression: true
                    }
                });
                
                const options = { mimeType: 'audio/webm' };
                this.recognitionMediaRecorder = new MediaRecorder(stream, options);
                this.recognitionAudioChunks = [];
                
                this.recognitionMediaRecorder.ondataavailable = (event) => {
                    if (event.data.size > 0) {
                        this.recognitionAudioChunks.push(event.data);
                    }
                };
                
                this.recognitionMediaRecorder.onstop = async () => {
                    const audioBlob = new Blob(this.recognitionAudioChunks, { type: 'audio/webm' });
                    
                    try {
                        const wavBlob = await this.convertToWav(audioBlob);
                        this.recognitionRecordedBlob = wavBlob;
                        this.hasRecognitionRecording = true;
                        this.selectedFile = null;
                        if (this.$refs.fileInput) {
                            this.$refs.fileInput.value = '';
                        }
                    } catch (error) {
                        console.error('音频转换失败:', error);
                        this.recognitionRecordingError = '音频转换失败，请重试';
                        this.hasRecognitionRecording = false;
                    }
                    
                    stream.getTracks().forEach(track => track.stop());
                };
                
                this.recognitionMediaRecorder.start();
                this.isRecognitionRecording = true;
                
            } catch (error) {
                console.error('麦克风访问失败:', error);
                if (error.name === 'NotAllowedError') {
                    this.recognitionRecordingError = '麦克风权限被拒绝';
                } else if (error.name === 'NotFoundError') {
                    this.recognitionRecordingError = '未找到麦克风设备';
                } else {
                    this.recognitionRecordingError = `麦克风访问失败: ${error.message}`;
                }
            }
        },
        stopRecognitionRecording() {
            if (this.recognitionMediaRecorder && this.recognitionMediaRecorder.state !== 'inactive') {
                this.recognitionMediaRecorder.stop();
                this.isRecognitionRecording = false;
            }
        },
        // ====== Clone Recording Methods ======
        async toggleCloneRecording() {
            if (this.isCloneRecording) {
                this.stopCloneRecording();
            } else {
                await this.startCloneRecording();
            }
        },
        async startCloneRecording() {
            try {
                this.cloneRecordingError = null;
                
                const stream = await navigator.mediaDevices.getUserMedia({ 
                    audio: {
                        channelCount: 1,
                        sampleRate: 16000,
                        echoCancellation: true,
                        noiseSuppression: true
                    }
                });
                
                const options = { mimeType: 'audio/webm' };
                this.cloneMediaRecorder = new MediaRecorder(stream, options);
                this.cloneAudioChunks = [];
                
                this.cloneMediaRecorder.ondataavailable = (event) => {
                    if (event.data.size > 0) {
                        this.cloneAudioChunks.push(event.data);
                    }
                };
                
                this.cloneMediaRecorder.onstop = async () => {
                    const audioBlob = new Blob(this.cloneAudioChunks, { type: 'audio/webm' });
                    
                    try {
                        const wavBlob = await this.convertToWav(audioBlob);
                        this.cloneRecordedBlob = wavBlob;
                        this.hasCloneRecording = true;
                        this.cloneReferenceFile = null;
                        if (this.$refs.cloneReferenceInput) {
                            this.$refs.cloneReferenceInput.value = '';
                        }
                    } catch (error) {
                        console.error('音频转换失败:', error);
                        this.cloneRecordingError = '音频转换失败，请重试';
                        this.hasCloneRecording = false;
                    }
                    
                    stream.getTracks().forEach(track => track.stop());
                };
                
                this.cloneMediaRecorder.start();
                this.isCloneRecording = true;
                
            } catch (error) {
                console.error('麦克风访问失败:', error);
                if (error.name === 'NotAllowedError') {
                    this.cloneRecordingError = '麦克风权限被拒绝';
                } else if (error.name === 'NotFoundError') {
                    this.cloneRecordingError = '未找到麦克风设备';
                } else {
                    this.cloneRecordingError = `麦克风访问失败: ${error.message}`;
                }
            }
        },
        stopCloneRecording() {
            if (this.cloneMediaRecorder && this.cloneMediaRecorder.state !== 'inactive') {
                this.cloneMediaRecorder.stop();
                this.isCloneRecording = false;
            }
        }
    }
}).mount('#app');