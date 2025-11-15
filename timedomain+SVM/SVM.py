import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc
from sklearn.preprocessing import label_binarize
import seaborn as sns
import pickle

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False


def svm_voice_recognition(feature_path='digit_features.csv'):
    """基于SVM实现孤立字语音识别，使用RBF核函数，生成综合AUC曲线"""
    # 加载特征数据
    if not os.path.exists(feature_path):
        raise FileNotFoundError(f"特征文件 {feature_path} 不存在，请先运行特征提取程序")
    df = pd.read_csv(feature_path)
    print(f"成功加载特征数据，共 {len(df)} 个样本，涵盖数字: {sorted(df['digit'].unique())}")

    # 特征选择与数据集划分
    feature_cols = [
        'ste_mean', 'ste_max', 'ste_std', 'ste_slope',
        'mn_mean', 'mn_max', 'mn_std', 'mn_crest',
        'zcr_mean', 'zcr_max', 'zcr_std', 'zcr_diff_mean',
        'f0_ac_mean', 'f0_ac_std', 'f0_am_mean', 'f0_am_std',
    ]
    X = df[feature_cols].values  # 特征矩阵
    y = df['digit'].values  # 数字类别标签

    # 分层抽样划分训练集、测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"数据集划分完成：训练集 {len(X_train)} 个样本，测试集 {len(X_test)} 个样本")

    # 特征标准化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 选择最优C值（正则化参数）
    best_C = 0.1
    best_acc = 0.0
    C_range = [0.1, 1, 10, 100, 200]
    acc_scores = []

    for C in C_range:
        # RBF核SVM，开启概率预测用于AUC计算
        svm = SVC(C=C, kernel='rbf', gamma='scale', probability=True, random_state=42)
        svm.fit(X_train_scaled, y_train)
        y_pred = svm.predict(X_test_scaled)
        acc = accuracy_score(y_test, y_pred)
        acc_scores.append(acc)
        if acc > best_acc:
            best_acc = acc
            best_C = C
    print(f"最优C值：{best_C}，测试集准确率：{best_acc:.2%}")

    # 最优模型训练与评估
    best_svm = SVC(C=best_C, kernel='rbf', gamma='scale', probability=True, random_state=42)
    best_svm.fit(X_train_scaled, y_train)
    y_pred = best_svm.predict(X_test_scaled)
    y_proba = best_svm.predict_proba(X_test_scaled)  # 概率值用于AUC计算

    # 输出评估指标
    print("\n===== 分类器详细评估 =====")
    print(f"准确率：{accuracy_score(y_test, y_pred):.2%}")
    print("\n混淆矩阵：")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    print("\n分类报告：")
    print(classification_report(y_test, y_pred))

    # 可视化1：C值与准确率关系
    plt.figure(figsize=(10, 6))
    plt.plot(C_range, acc_scores, marker='o', color='#ff7f0e')
    plt.axvline(x=best_C, color='r', linestyle='--', label=f'最优C={best_C}')
    plt.xscale('log')  # 对数刻度更直观
    plt.xlabel('C值（正则化参数）')
    plt.ylabel('准确率')
    plt.title('不同C值对SVM分类器性能的影响（RBF核）')
    plt.grid(alpha=0.3)
    plt.legend()
    plt.savefig('svm_c_accuracy.png', dpi=300)
    plt.show()

    # 可视化2：混淆矩阵热图
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges',
                xticklabels=[f'{i}' for i in range(10)],
                yticklabels=[f'{i}' for i in range(10)])
    plt.xlabel('预测标签')
    plt.ylabel('真实标签')
    plt.title('SVM分类器混淆矩阵')
    plt.tight_layout()
    plt.savefig('svm_confusion_matrix.png', dpi=300)
    plt.show()

    # 可视化3：宏平均AUC曲线
    y_test_binarized = label_binarize(y_test, classes=range(10))
    n_classes = y_test_binarized.shape[1]

    # 计算各类别ROC曲线
    fpr = dict()
    tpr = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_binarized[:, i], y_proba[:, i])

    # 计算宏平均ROC
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    mean_tpr /= n_classes  # 平均TPR
    roc_auc_macro = auc(all_fpr, mean_tpr)

    # 绘制AUC曲线（与KNN样式统一）
    plt.figure(figsize=(6, 6))
    plt.plot(all_fpr, mean_tpr, 'orange', linewidth=2, label='ROC曲线')
    plt.fill_between(all_fpr, mean_tpr, alpha=0.2, color='orange', label='曲线下面积(AUC)')
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1.5)  # 随机基准线

    # 标记最佳工作点
    gmeans = np.sqrt(mean_tpr * (1 - all_fpr))
    best_idx = np.argmax(gmeans)
    work_fpr, work_tpr = all_fpr[best_idx], mean_tpr[best_idx]
    plt.scatter(work_fpr, work_tpr, color='r', s=60, zorder=5, label='工作点')
    plt.text(work_fpr + 0.03, work_tpr - 0.08,
             f'({work_fpr:.2f},{work_tpr:.2f})', color='r', fontsize=10)

    # 添加AUC值与标签
    plt.text(0.4, 0.3, f'AUC = {roc_auc_macro:.3f}', color='black', fontsize=12, fontweight='bold')
    plt.xlabel('假正率', fontsize=10)
    plt.ylabel('真正率', fontsize=10)
    plt.title('SVM模型', fontsize=12, fontweight='bold')
    plt.legend(loc='lower right', frameon=True, facecolor='white', edgecolor='gray')
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.savefig('svm_macro_auc.png', dpi=300, bbox_inches='tight')
    plt.show()

    print(f"\n宏平均AUC值：{roc_auc_macro:.3f}")
    return best_svm, scaler, best_acc


if __name__ == "__main__":
    svm_model, scaler, accuracy = svm_voice_recognition('digit_features.csv')
    print("\nSVM分类器训练完成，已生成混淆矩阵和综合AUC曲线")

    with open('svm_model.pkl', 'wb') as f:
        pickle.dump(svm_model, f)
    with open('svm_scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    print("模型已保存为 svm_model.pkl 和 svm_scaler.pkl")