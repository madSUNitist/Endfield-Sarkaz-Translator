import os
import re

DATASET_DIR = "corpus/datasets"
RAW_FILE = os.path.join(DATASET_DIR, "raw_zh.txt")
SRC_FILE = os.path.join(DATASET_DIR, "train.src")
TGT_FILE = os.path.join(DATASET_DIR, "train.tgt")

# 替换为游戏真实的 56 字符表
CHARACTER_MAP = "abcdefghijklmnopqrstuvwxyz0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"

def encode(text):
    return "".join(CHARACTER_MAP[ord(c) % 56] if c not in "\n\t\r" else c for c in text)

def process():
    with open(RAW_FILE, "r", encoding="utf-8") as f_in, \
         open(SRC_FILE, "w", encoding="utf-8") as f_src, \
         open(TGT_FILE, "w", encoding="utf-8") as f_tgt:
        
        for line in f_in:
            txt = re.sub(r"\s+", " ", line.strip())
            if 10 <= len(txt) <= 200:
                f_src.write(txt + "\n")
                f_tgt.write(encode(txt) + "\n")
    print("处理完成")

if __name__ == "__main__":
    process()