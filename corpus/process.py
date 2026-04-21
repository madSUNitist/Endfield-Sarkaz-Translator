import os
import re
from dotenv import load_dotenv
from datasets import load_dataset
from tqdm import tqdm

load_dotenv()

# 1. 全量下载至本地缓存（首次约 1.2GB，后续直接读缓存秒开）
ds = load_dataset(
    "wikimedia/wikipedia",
    "20231101.zh",
    split="train"
)

CHARACTER_MAP = "gkamztlbdqiyfucxbhsjoprnweygtjmevchdxsanqolkrvwiypjzquhe"

def encode_line(text: str) -> str:
    return "".join(CHARACTER_MAP[ord(c) % 56] if c not in "\n\t\r" else c for c in text)

MAX_CHARS = float('inf')
src_path = "corpus/datasets/train.zh"
tgt_path = "corpus/datasets/train.skz"
os.makedirs("corpus/datasets", exist_ok=True)
total_chars = 0

with open(src_path, "w", encoding="utf-8") as f_src, \
     open(tgt_path, "w", encoding="utf-8") as f_tgt:
    
    # 此时 ds 为标准 Dataset，支持快速迭代
    for item in tqdm(ds, desc="处理语料", unit="item"):
        txt = re.sub(r"\s+", " ", item["text"]).strip()
        if len(txt) < 20 or len(txt) > 200:
            continue

        f_src.write(txt + "\n")
        f_tgt.write(encode_line(txt) + "\n")
        
        total_chars += len(txt)
        if total_chars >= MAX_CHARS:
            break

print(f"Generate complete: {os.path.getsize(src_path)/1e6:.1f}MB / {os.path.getsize(tgt_path)/1e6:.1f}MB")