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

# 创建模型保存目录
os.makedirs("model", exist_ok=True)

# 1. 读取本地 CSV 文件（放在项目根目录）
csv_path = "imdb_top_500.csv"
if not os.path.exists(csv_path):
    raise FileNotFoundError(f"请确保 {csv_path} 文件存在于项目根目录")

df = pd.read_csv(csv_path)

# 2. 自动识别文本列和标签列
possible_text_cols = ['review', 'text', 'comment', 'content', 'review_text']
possible_label_cols = ['sentiment', 'label', 'class', 'polarity']

text_col = None
label_col = None

for col in possible_text_cols:
    if col in df.columns:
        text_col = col
        break
for col in possible_label_cols:
    if col in df.columns:
        label_col = col
        break

if text_col is None or label_col is None:
    raise ValueError(f"无法识别列名。请检查 CSV 的列：{list(df.columns)}")

print(f"使用文本列: {text_col}, 标签列: {label_col}")

# 3. 将标签转换为 0/1（正面=1，负面=0）
# 常见取值: positive/neg, 1/0, pos/neg, +1/-1 等
def to_binary_label(val):
    val_str = str(val).lower()
    if val_str in ['positive', 'pos', '1', '1.0', 'good', 'yes']:
        return 1
    else:
        return 0

df['label'] = df[label_col].apply(to_binary_label)

# 4. 读取配置
with open("config.json", "r") as f:
    config = json.load(f)

max_features = config["max_features"]
ngram_range = tuple(config["ngram_range"])
test_size = config["test_size"]
batch_size = config["batch_size"]
epochs = config["epochs"]
lr = config["learning_rate"]

# 5. TF-IDF 向量化
print("正在进行 TF-IDF 向量化...")
vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=ngram_range)
X = vectorizer.fit_transform(df[text_col].fillna('')).toarray()
y = df['label'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

# 转换为 PyTorch Tensor
X_train_t = torch.tensor(X_train, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.long)
X_test_t = torch.tensor(X_test, dtype=torch.float32)
y_test_t = torch.tensor(y_test, dtype=torch.long)

# 6. 定义神经网络
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

# 7. 训练函数
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

# 8. 评估
model.eval()
with torch.no_grad():
    logits = model(X_test_t)
    preds = torch.argmax(logits, dim=1).numpy()
acc = accuracy_score(y_test, preds)
f1 = f1_score(y_test, preds)
print(f"Test Accuracy: {acc:.4f}, F1 Score: {f1:.4f}")

# 9. 保存指标
metrics = {
    "accuracy": float(acc),
    "f1_score": float(f1),
    "epochs": epochs,
    "batch_size": batch_size,
    "learning_rate": lr,
    "max_features": max_features
}
with open("model/metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

# 10. 保存模型和向量器
torch.save(model.state_dict(), "model/model.pt")
with open("model/vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

print("训练完成。所有产物已保存到 model/ 目录")