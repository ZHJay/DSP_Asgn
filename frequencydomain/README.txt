Transformers.py        （注意改成自己的路径，第一次运行清按照顺序一步一步来）
启动：python transformers.py --audio_root "/Users/zhanghjay/Desktop/DSP_Asgn/frequencydomain/样本"

可视化-完整模型：python visualize.py --snapshot outputs/embedding_snapshot.pt --attn outputs/attention_weights.pt


频域处理：python predict_single.py --audio "/Users/zhanghjay/Desktop/DSP_Asgn/frequencydomain/样本/3/bck-3-3.wav"       -------注意这里修改一下测试的wav

消融实验：
# 1) FULL
python train_ablate.py --audio_root "/Users/zhanghjay/Desktop/DSP/frequencydomain/样本" --ablation full

# 2) NO_MEL
python train_ablate.py --audio_root "/Users/zhanghjay/Desktop/DSP/frequencydomain/样本" --ablation no_mel

# 3) NO_W2V2
python train_ablate.py --audio_root "/Users/zhanghjay/Desktop/DSP/frequencydomain/样本" --ablation no_w2v2

# 4) RAW BASELINE
python train_ablate.py --audio_root "/Users/zhanghjay/Desktop/DSP/frequencydomain/样本" --ablation raw

#enhanced
python train_enhanced.py --audio_root "/Users/zhanghjay/Desktop/DSP/frequencydomain/样本"

消融实验对比绘图
python plot_ablation.py

智能锁：
python smart_lock.py --audio "/Users/zhanghjay/Desktop/DSP_Asgn/frequencydomain/lock3.wav" --model "outputs/best_model.pt" --metrics "outputs/metrics.json"
---------正确密码-------------
python smart_lock.py --audio "/Users/zhanghjay/Desktop/DSP_Asgn/frequencydomain/lock1.wav" --model "outputs/best_model.pt" --metrics "outputs/metrics.json"
--------错误密码-------------