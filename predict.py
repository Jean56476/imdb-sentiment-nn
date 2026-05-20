import pickle
import torch
import json

class SentimentNN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dims=[128,64], dropout=0.3):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.append(torch.nn.Linear(prev, h))
            layers.append(torch.nn.ReLU())
            layers.append(torch.nn.Dropout(dropout))
            prev = h
        layers.append(torch.nn.Linear(prev, 2))
        self.net = torch.nn.Sequential(*layers)
    def forward(self, x):
        return self.net(x)

def load_model(model_path="model/model.pt", vec_path="model/vectorizer.pkl", cfg_path="config.json"):
    with open(vec_path, "rb") as f:
        vec = pickle.load(f)
    with open(cfg_path) as f:
        cfg = json.load(f)
    model = SentimentNN(cfg["max_features"])
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    return model, vec

def predict(text, model, vec):
    x = vec.transform([text]).toarray()
    t = torch.tensor(x, dtype=torch.float32)
    with torch.no_grad():
        logits = model(t)
        prob = torch.softmax(logits, dim=1)[0,1].item()
    return "positive" if prob>0.5 else "negative", prob

if __name__ == "__main__":
    m, v = load_model()
    while True:
        txt = input("Review: ")
        if txt == "quit": break
        sent, conf = predict(txt, m, v)
        print(f"{sent} ({conf:.3f})")