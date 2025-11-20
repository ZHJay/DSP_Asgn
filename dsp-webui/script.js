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
    },
    methods: {
        onFileChange(event) {
            const [file] = event.target.files || [];
            this.selectedFile = file || null;

            if (file) {
                this.lastProcessedFile = null;
                this.resetAllConfidenceTemplates();
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
            if (!this.selectedFile) {
                alert('请上传一个 WAV 文件！');
                return;
            }

            this.appendMessage('正在上传并预处理，请稍候...');
            this.isUploading = true;

            const formData = new FormData();
            formData.append('wavFile', this.selectedFile);

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
        },
        appendCloneMessage(message) {
            this.cloneMessages.push(message);
        },
        async handleClone() {
            if (!this.cloneReferenceFile) {
                alert('请选择参考音频文件！');
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
            formData.append('referenceAudio', this.cloneReferenceFile);
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
        }
    }
}).mount('#app');