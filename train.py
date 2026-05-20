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
from urllib.request import urlretrieve

# 创建目录
os.makedirs("data", exist_ok=True)
os.makedirs("model", exist_ok=True)

# 下载数据集（如果本地没有）
data_url = "https://raw.githubusercontent.com/EmporioSabo/imdb-sentiment-nn/main/data/imdb_balanced_10k.csv"
data_path = "data/imdb_balanced_10k.csv"

if not os.path.exists(data_path):
    print("Downloading dataset...")
    urlretrieve(data_url, data_path)

df = pd.read_csv(data_path)
# 确保有 'review' 和 'sentiment' 列。如果没有，尝试调整
if 'sentiment' not in df.columns:
    # 有的数据集列名叫 'label' 或 'class'
    if 'label' in df.columns:
        df.rename(columns={'label': 'sentiment'}, inplace=True)
    elif 'class' in df.columns:
        df.rename(columns={'class': 'sentiment'}, inplace=True)

df['label'] = df['sentiment'].apply(lambda x: 1 if str(x).lower() in ['positive', 'pos', 1] else 0)

# 读取配置
with open("config.json", "r") as f:
    config = json.load(f)

max_features = config["max_features"]
ngram_range = tuple(config["ngram_range"])
test_size = config["test_size"]
batch_size = config["batch_size"]
epochs = config["epochs"]
lr = config["learning_rate"]

# TF-IDF 向量化
vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=ngram_range)
X = vectorizer.fit_transform(df['review']).toarray()
y = df['label'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

# 转为 Tensor
X_train_t = torch.tensor(X_train, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.long)
X_test_t = torch.tensor(X_test, dtype=torch.float32)
y_test_t = torch.tensor(y_test, dtype=torch.long)

# 定义神经网络
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

# 训练函数
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
for epoch in range(epochs):
    loss = train_epoch(model, X_train_t, y_train_t, optimizer, criterion, batch_size)
    print(f"Epoch {epoch+1}/{epochs}, Loss: {loss:.4f}")

# 评估
model.eval()
with torch.no_grad():
    logits = model(X_test_t)
    preds = torch.argmax(logits, dim=1).numpy()
acc = accuracy_score(y_test, preds)
f1 = f1_score(y_test, preds)
print(f"Test Acc: {acc:.4f}, F1: {f1:.4f}")

# 保存指标
metrics = {"accuracy": acc, "f1_score": f1, "epochs": epochs, "batch_size": batch_size}
with open("model/metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

# 保存模型和向量器
torch.save(model.state_dict(), "model/model.pt")
with open("model/vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

print("Training finished. Artifacts saved in model/")