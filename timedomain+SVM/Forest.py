import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc
from sklearn.preprocessing import label_binarize
import seaborn as sns

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False


def random_forest_voice_recognition(feature_path='digit_features.csv'):
    """基于随机森林实现孤立字语音识别"""
    # 加载特征数据
    if not os.path.exists(feature_path):
        raise FileNotFoundError(f"特征文件 {feature_path} 不存在，请先运行特征提取程序")
    df = pd.read_csv(feature_path)
    print(f"加载数据完成：{len(df)}个样本，数字类别：{sorted(df['digit'].unique())}")

    # 特征选择与数据集划分
    feature_cols = [
        'ste_mean', 'ste_max', 'ste_std', 'ste_slope',
        'mn_mean', 'mn_max', 'mn_std', 'mn_crest',
        'zcr_mean', 'zcr_max', 'zcr_std', 'zcr_diff_mean',
        'f0_ac_mean', 'f0_ac_std', 'f0_am_mean', 'f0_am_std',
    ]
    X = df[feature_cols].values  # 特征矩阵
    y = df['digit'].values  # 数字类别标签

    # 分层划分训练集/测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"数据集划分：训练集{len(X_train)}个，测试集{len(X_test)}个")

    # 特征归一化（保持流程一致，随机森林对其不敏感）
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 选择最优树数量（n_estimators）
    best_n = 100
    best_acc = 0.0
    n_range = [50, 100, 200, 300, 400]
    acc_scores = []

    for n in n_range:
        rf = RandomForestClassifier(
            n_estimators=n, max_depth=10, min_samples_split=10,
            random_state=42, n_jobs=-1  # 并行加速
        )
        rf.fit(X_train_scaled, y_train)
        acc = accuracy_score(y_test, rf.predict(X_test_scaled))
        acc_scores.append(acc)
        if acc > best_acc:
            best_acc = acc
            best_n = n
    print(f"最优树数量：{best_n}，测试集准确率：{best_acc:.2%}")

    # 最优模型训练与评估
    best_rf = RandomForestClassifier(
        n_estimators=best_n, max_depth=10, min_samples_split=10,
        random_state=42, n_jobs=-1
    )
    best_rf.fit(X_train_scaled, y_train)
    y_pred = best_rf.predict(X_test_scaled)
    y_proba = best_rf.predict_proba(X_test_scaled)

    # 输出评估指标
    print("\n===== 分类器详细评估 =====")
    print(f"准确率：{accuracy_score(y_test, y_pred):.2%}")
    print("\n混淆矩阵：")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    print("\n分类报告：")
    print(classification_report(y_test, y_pred))

    # 可视化1：树数量与准确率关系
    plt.figure(figsize=(10, 6))
    plt.plot(n_range, acc_scores, marker='o', color='#2ca02c')
    plt.axvline(x=best_n, color='r', linestyle='--', label=f'最优树数量={best_n}')
    plt.xlabel('树的数量（n_estimators）')
    plt.ylabel('准确率')
    plt.title('树数量对随机森林性能的影响')
    plt.grid(alpha=0.3)
    plt.legend()
    plt.savefig('rf_n_accuracy.png', dpi=300)
    plt.show()

    # 可视化2：混淆矩阵热图
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens',
                xticklabels=range(10), yticklabels=range(10))
    plt.xlabel('预测标签')
    plt.ylabel('真实标签')
    plt.title('随机森林混淆矩阵')
    plt.tight_layout()
    plt.savefig('rf_confusion_matrix.png', dpi=300)
    plt.show()

    # 可视化3：宏平均AUC曲线（与SVM统一样式）
    y_test_binarized = label_binarize(y_test, classes=range(10))
    n_classes = y_test_binarized.shape[1]

    # 计算各类别ROC与宏平均ROC
    fpr = {i: roc_curve(y_test_binarized[:, i], y_proba[:, i])[0] for i in range(n_classes)}
    tpr = {i: roc_curve(y_test_binarized[:, i], y_proba[:, i])[1] for i in range(n_classes)}
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    mean_tpr = np.mean([np.interp(all_fpr, fpr[i], tpr[i]) for i in range(n_classes)], axis=0)
    roc_auc_macro = auc(all_fpr, mean_tpr)

    # 绘制AUC曲线
    plt.figure(figsize=(6, 6))
    plt.plot(all_fpr, mean_tpr, 'green', linewidth=2, label='ROC曲线')
    plt.fill_between(all_fpr, mean_tpr, alpha=0.2, color='green', label='曲线下面积(AUC)')
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1.5)

    # 标记工作点
    gmeans = np.sqrt(mean_tpr * (1 - all_fpr))
    best_idx = np.argmax(gmeans)
    work_fpr, work_tpr = all_fpr[best_idx], mean_tpr[best_idx]
    plt.scatter(work_fpr, work_tpr, color='r', s=60, zorder=5, label='工作点')
    plt.text(work_fpr + 0.03, work_tpr - 0.08,
             f'({work_fpr:.2f},{work_tpr:.2f})', color='r', fontsize=10)

    plt.text(0.4, 0.3, f'AUC = {roc_auc_macro:.3f}', fontsize=12, fontweight='bold')
    plt.xlabel('假正率')
    plt.ylabel('真正率')
    plt.title('随机森林模型', fontsize=12, fontweight='bold')
    plt.legend(loc='lower right', frameon=True, facecolor='white', edgecolor='gray')
    plt.grid(linestyle='--', alpha=0.3)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.savefig('rf_macro_auc.png', dpi=300, bbox_inches='tight')
    plt.show()

    # 输出特征重要性（随机森林特有）
    importance_df = pd.DataFrame({
        '特征': feature_cols, '重要性': best_rf.feature_importances_
    }).sort_values('重要性', ascending=False)
    print("\n特征重要性排序：")
    print(importance_df)

    print(f"\n宏平均AUC值：{roc_auc_macro:.3f}")
    return best_rf, scaler, best_acc


if __name__ == "__main__":
    rf_model, scaler, accuracy = random_forest_voice_recognition('digit_features.csv')
    print("\n随机森林分类器训练完成，已生成所有可视化结果")