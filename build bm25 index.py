"""
build_bm25_index.py
--------------------
基于 MITRE ATT&CK 文档构建 BM25 关键词索引，供混合检索使用。
BM25 擅长精确关键词匹配，弥补纯向量检索的语义漂移问题。

依赖：
    pip install rank-bm25

使用方式：
    python build_bm25_index.py
"""

import json
import pickle
import re
from pathlib import Path

import pandas as pd
from rank_bm25 import BM25Okapi

DATASET2_PATH  = "./data/input/MITRE-ATTACK_dataset_test.json"
BM25_INDEX_PATH = "./data/database/ttp_bm25_index.pkl"


def tokenize(text: str) -> list:
    """
    简单的英文分词：小写 + 按非字母数字切分。
    保留连字符词（如 command-and-control）作为整体。
    """
    text = text.lower()
    tokens = re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)*", text)
    return tokens


def build():
    df = pd.read_json(DATASET2_PATH, orient="records")

    docs = []
    for _, row in df.iterrows():
        desc = row.get("description", "")
        if not isinstance(desc, str) or not desc.strip():
            continue
        docs.append({
            "id":          row["ID"],
            "name":        row["name"],
            "description": desc,
        })

    print(f"[+] 加载 {len(docs)} 条 TTP 文档")

    # 拼接 name + description 一起建索引
    # name 里有明确的技术名称（如 "OS Credential Dumping"），对召回帮助很大
    corpus_tokens = [
        tokenize(doc["name"] + " " + doc["description"])
        for doc in docs
    ]

    bm25 = BM25Okapi(corpus_tokens)

    Path(BM25_INDEX_PATH).parent.mkdir(parents=True, exist_ok=True)
    with open(BM25_INDEX_PATH, "wb") as f:
        pickle.dump({"bm25": bm25, "docs": docs}, f)

    print(f"[+] BM25 索引已保存到 {BM25_INDEX_PATH}")
    print(f"    词汇表大小：{len(bm25.idf)} 个词")


if __name__ == "__main__":
    build()