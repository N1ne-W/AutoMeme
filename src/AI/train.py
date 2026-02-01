import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

# -----------------------------
# 1. 数据加载函数
# -----------------------------
def load_dataset(base_path, label):
    print("\n正在加载:", base_path)
    print("路径是否存在:", os.path.exists(base_path))

    X = []
    y = []

    if not os.path.exists(base_path):
        print("❌ 路径不存在")
        return np.array([]), np.array([])

    for file in os.listdir(base_path):
        if file.endswith(".npy"):
            file_path = os.path.join(base_path, file)
            data = np.load(file_path)
            X.append(data)
            y.append(label)

    print(f"{base_path} 加载完成：{len(X)} 条样本")
    return np.array(X), np.array(y)



# -----------------------------
# 2. 路径配置（一定要和你的真实目录一致）
# 你的结构应该是：
# AutoMeme/
#   dataset/
#     donk/
#       sample_xxx/
#     monkeythink/
#       sample_xxx/
# -----------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DONK_PATH = r"/dataset/donk"
MONKEY_PATH = r"/dataset/monkeythink"

# -----------------------------
# 3. 加载数据
# -----------------------------
X_donk, y_donk = load_dataset(DONK_PATH, 0)
X_monkey, y_monkey = load_dataset(MONKEY_PATH, 1)

if len(X_donk) == 0 and len(X_monkey) == 0:
    print("\n❌ 没有加载到任何数据，请先检查数据路径和目录结构")

# -----------------------------
# 4. 合并数据
# -----------------------------
X = np.vstack([X_donk, X_monkey])
y = np.concatenate([y_donk, y_monkey])

print("\n总样本数:", len(X))
print("特征维度:", X.shape)

# -----------------------------
# 5. 切分训练集和测试集
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, shuffle=True
)

print("训练集数量:", len(X_train))
print("测试集数量:", len(X_test))

# -----------------------------
# 6. 建立模型
# -----------------------------
model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

# -----------------------------
# 7. 训练模型
# -----------------------------
print("\n开始训练模型...")
model.fit(X_train, y_train)

# -----------------------------
# 8. 测试模型
# -----------------------------
y_pred = model.predict(X_test)

acc = accuracy_score(y_test, y_pred)
print("\n准确率:", acc)
print(classification_report(y_test, y_pred, target_names=["donk", "monkey"]))

# -----------------------------
# 9. 保存模型
# -----------------------------
joblib.dump(model, r"/first_model.pkl")
print("\n模型已保存为 first_model.pkl")
