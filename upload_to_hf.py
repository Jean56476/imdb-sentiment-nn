import os
from huggingface_hub import HfApi

def main():
    token = os.environ["HF_TOKEN"]
    repo_id = "Jean56476/imdb-sentiment-nn"   # 替换成你的用户名/仓库名
    api = HfApi()
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