import os
import json
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

# 创建目录
os.makedirs("model", exist_ok=True)

# 1. 读取 CSV 文件（根据你提供的列名）
csv_path = "imdb_top_500.csv"
if not os.path.exists(csv_path):
    raise FileNotFoundError(f"请确保 {csv_path} 文件存在于项目根目录")

df = pd.read_csv(csv_path)

# 你的数据列名是: text, label, rating
# 将 'text' 作为评论文本，'label' 作为标签（已经是 0/1）
if 'text' not in df.columns or 'label' not in df.columns:
    raise ValueError(f"CSV 文件缺少 'text' 或 'label' 列，实际列名: {list(df.columns)}")

print(f"数据加载成功，共 {len(df)} 条记录")
print(f"标签分布:\n{df['label'].value_counts()}")

# 2. 配置参数（如果 config.json 不存在则使用默认值）
config_path = "config.json"
if os.path.exists(config_path):
    with open(config_path, "r") as f:
        config = json.load(f)
else:
    # 默认配置
    config = {
        "max_features": 5000,
        "ngram_range": [1, 2],
        "test_size": 0.2,
        "batch_size": 64,
        "epochs": 10,
        "learning_rate": 0.001
    }
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print("已创建默认 config.json")

max_features = config["max_features"]
ngram_range = tuple(config["ngram_range"])
test_size = config["test_size"]
batch_size = config["batch_size"]
epochs = config["epochs"]
lr = config["learning_rate"]

# 3. TF-IDF 向量化
print("正在进行 TF-IDF 向量化...")
vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=ngram_range)
X = vectorizer.fit_transform(df['text'].fillna('')).toarray()
y = df['label'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

# 转换为 PyTorch Tensor
X_train_t = torch.tensor(X_train, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.long)
X_test_t = torch.tensor(X_test, dtype=torch.float32)
y_test_t = torch.tensor(y_test, dtype=torch.long)

# 4. 定义神经网络
class SentimentNN(nn.Module):
    def __init__(self, input_dim, hidden_dims=[128, 64], dropout=0.3):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, 2))
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        return self.net(x)

model = SentimentNN(input_dim=X_train.shape[1])
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

# 5. 训练函数
def train_epoch(model, X, y, optimizer, criterion, batch_size):
    model.train()
    idx = np.random.permutation(len(X))
    total_loss = 0
    for i in range(0, len(X), batch_size):
        batch_idx = idx[i:i+batch_size]
        Xb = X[batch_idx]
        yb = y[batch_idx]
        optimizer.zero_grad()
        out = model(Xb)
        loss = criterion(out, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / (len(X)//batch_size + 1)

# 训练循环
print("开始训练...")
for epoch in range(epochs):
    loss = train_epoch(model, X_train_t, y_train_t, optimizer, criterion, batch_size)
    print(f"Epoch {epoch+1}/{epochs}, Loss: {loss:.4f}")

# 6. 评估
model.eval()
with torch.no_grad():
    logits = model(X_test_t)
    preds = torch.argmax(logits, dim=1).numpy()
acc = accuracy_score(y_test, preds)
f1 = f1_score(y_test, preds)
print(f"测试集准确率: {acc:.4f}, F1分数: {f1:.4f}")

# 7. 保存指标
metrics = {
    "accuracy": float(acc),
    "f1_score": float(f1),
    "epochs": epochs,
    "batch_size": batch_size,
    "learning_rate": lr,
    "max_features": max_features,
    "dataset_size": len(df)
}
with open("model/metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

# 8. 保存模型和向量器
torch.save(model.state_dict(), "model/model.pt")
with open("model/vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

print("训练完成。所有产物已保存到 model/ 目录")