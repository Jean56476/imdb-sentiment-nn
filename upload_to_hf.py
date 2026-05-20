import os
from huggingface_hub import HfApi

def main():
    token = os.environ["HF_TOKEN"]
    repo_id = "Jean256/imdb-sentiment-nn"   # 改成你的真实 Hugging Face 用户名
    api = HfApi()

    # 如果 repo 不存在则创建
    api.create_repo(repo_id, token=token, exist_ok=True)

    artifacts = [
        "model/model.pt",
        "model/vectorizer.pkl",
        "config.json",
        "model/metrics.json"
    ]

    for f in artifacts:
        if not os.path.exists(f):
            print(f"Warning: {f} not found, skipping")
            continue
        name = f.split("/")[-1]
        api.upload_file(
            path_or_fileobj=f,
            path_in_repo=name,
            repo_id=repo_id,
            token=token,
        )
        print(f"Uploaded {f}")

if __name__ == "__main__":
    main()