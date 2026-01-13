import pandas as pd
import numpy as np
import pywt
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

# 设置字体与负号显示
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['axes.unicode_minus'] = False

# ===================== 数据读取 =====================
def process_signal_data(folder_path, initial_file, start=150):
    signals = pd.read_csv(initial_file)
    files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
    for file in files[1:]:
        temp_df = pd.read_csv(os.path.join(folder_path, file))
        signal_column = temp_df.columns[1]
        signals[file] = temp_df[signal_column]
    signals = signals.loc[start:start + 2000 - 1]
    signals['in s'] = np.arange(0, 0.5 * len(signals), 0.5)
    signals_transposed = signals.T
    signals_transposed.columns = signals_transposed.iloc[0].values
    signals_transposed = signals_transposed.drop(['in s'])
    signals_transposed.reset_index(drop=True, inplace=True)
    return signals_transposed

# ===================== 小波特征提取（A8 + D8~D1  = 9维） =====================
def extract_wavelet_features(signals, wavelet='db4', level=8, include_entropy=True):
    features = []
    for i in range(signals.shape[0]):
        signal = signals.iloc[i, :].values
        coeffs = pywt.wavedec(signal, wavelet=wavelet, level=level)
        energy = [np.sum(c ** 2) for c in coeffs]  # A8, D8, D7, ..., D1

        selected_energy = energy  # 9维能量

        if include_entropy:
            probs = np.array(energy) / np.sum(energy)
            entropy = -np.sum([p * np.log2(p) for p in probs if p > 0])
            feature_vector = selected_energy + [entropy]  # 共10维
        else:
            feature_vector = selected_energy

        features.append(feature_vector)
    return np.array(features)

# ===================== 加载信号数据 =====================
air_signals = process_signal_data('voice/air', 'voice/air/air (1).csv')
breathe_signals = process_signal_data('voice/breathe', 'voice/breathe/breathe (1).csv')
combustion_signals = process_signal_data('voice/combustion', 'voice/combustion/combustion (1).csv')
dryice_signals = process_signal_data('voice/dryice', 'voice/dryice/dryice (1).csv')

# ===================== 提取小波特征（9维） =====================
air_features = extract_wavelet_features(air_signals, level=8, include_entropy=True)
breathe_features = extract_wavelet_features(breathe_signals, level=8, include_entropy=True)
combustion_features = extract_wavelet_features(combustion_signals, level=8, include_entropy=True)
dryice_features = extract_wavelet_features(dryice_signals, level=8, include_entropy=True)

features = np.vstack([air_features, breathe_features, combustion_features, dryice_features])
labels = np.concatenate([
    np.zeros(air_features.shape[0]),
    np.ones(breathe_features.shape[0]),
    np.full(combustion_features.shape[0], 2),
    np.full(dryice_features.shape[0], 3)
])

# ===================== 打印特征维度 =====================
print("最终特征维度:", features.shape[1])  # 应该是10
print("示例特征向量:", air_features[0])

# ===================== 划分数据集 =====================
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2,
                                                    random_state=42, stratify=labels)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ===================== 逻辑回归分类并评估 =====================
model = LogisticRegression(max_iter=10000, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# 总体评估
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

print("\n📊 Overall Performance (Logistic Regression):")
print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1-score:  {f1:.4f}")

# 每类识别度（精度、召回率、F1）
target_names = ["air", "breathe", "combustion", "dryice"]
report = classification_report(y_test, y_pred, target_names=target_names, digits=4)
print("\n📌 Per-class Classification Report:")
print(report)

