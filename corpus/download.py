import os
import urllib.request

DATASET_DIR = "corpus/datasets"
RAW_FILE = os.path.join(DATASET_DIR, "raw_zh.txt")
URL = "https://huggingface.co/datasets/wikimedia/wikipedia/resolve/main/data/zh/20231101/zh-train.txt"

def download():
    os.makedirs(DATASET_DIR, exist_ok=True)
    if os.path.exists(RAW_FILE):
        print("文件已存在")
        return
    print("开始下载...")
    urllib.request.urlretrieve(URL, RAW_FILE)
    print("下载完成")

if __name__ == "__main__":
    download()