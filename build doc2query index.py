"""
build_doc2query_index.py
------------------------
Doc2Query 建库策略：在索引阶段用 LLM 为每条 TTP 生成合成 CTI 查询句子，
将合成句子的 embedding 存入向量库，命中后返回原始 TTP 父文档。

核心思路：
  - 真实 CTI 句子（行为语言）↔ 合成 CTI 句子（行为语言）→ 向量空间接近，召回精准
  - 合成句子作为 child，原始 TTP description 作为 parent，结构与 ParentDocumentRetriever 一致

建库流程：
  1. 读取 508 条 MITRE TTP
  2. 并发调用 Qwen，每条 TTP 生成 N 条合成 CTI 查询
  3. 合成查询存入 ChromaDB（child），原始文档存入 LocalFileStore（parent）
  4. 评估时直接替换检索器，其余代码不变

使用方式：
    python build_doc2query_index.py
    python build_doc2query_index.py --n_queries 5 --batch_size 20
    python build_doc2query_index.py --resume   # 跳过已生成的条目，断点续跑
"""

import os
import json
import time
import argparse
import hashlib
from pathlib import Path

import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv(Path(".") / ".env")

from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma
from langchain_community.embeddings import DashScopeEmbeddings
from langchain.storage import LocalFileStore, create_kv_docstore
from langchain_core.documents import Document

# ── 路径配置 ─────────────────────────────────────────────────────────────────
DATASET2_PATH       = "./data/input/MITRE-ATTACK_dataset_test.json"
D2Q_VECTORDB_DIR    = "./data/database/ttp_doc2query_chroma_db/"
D2Q_DOCSTORE_DIR    = "./data/database/ttp_doc2query_docstore/"
D2Q_CACHE_PATH      = "./data/database/doc2query_cache.json"   # 断点续跑缓存


# ────────────────────────────────────────────────────────────────────────────
# Prompt：指示 LLM 生成合成 CTI 查询
# ────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a cybersecurity expert specializing in threat intelligence.
Your task is to generate realistic CTI (Cyber Threat Intelligence) sentences that describe
a specific MITRE ATT&CK technique from the perspective of a threat analyst or malware report.

Rules:
- Each sentence must describe concrete adversary behavior, tools, or observed actions
- Use varied vocabulary: mix tool names, malware families, and generic descriptions
- Sentences should sound like excerpts from real threat reports
- Do NOT mention the technique name or ID directly
- Output ONLY a JSON array of strings, no explanation, no markdown
"""

def build_user_prompt(ttp_name: str, ttp_desc: str, n: int) -> str:
    # 只取 description 的前 600 字符，避免 prompt 过长
    short_desc = ttp_desc[:600].strip()
    return f"""Technique: {ttp_name}
Description (excerpt): {short_desc}

Generate exactly {n} diverse CTI sentences that a threat analyst might write when
reporting an adversary using this technique. Output as a JSON array."""


# ────────────────────────────────────────────────────────────────────────────
# LLM 调用：生成合成查询
# ────────────────────────────────────────────────────────────────────────────

def generate_synthetic_queries(
    llm: ChatOpenAI,
    ttp_id: str,
    ttp_name: str,
    ttp_desc: str,
    n: int,
) -> list[str]:
    """
    调用 Qwen 为单条 TTP 生成 n 条合成 CTI 查询。
    失败时返回空列表，不中断建库流程。
    """
    user_msg = build_user_prompt(ttp_name, ttp_desc, n)
    try:
        response = llm.invoke([
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_msg},
        ])
        raw = response.content.strip()

        # 清理可能的 markdown 代码块
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        raw = raw.strip()

        queries = json.loads(raw)
        if not isinstance(queries, list):
            raise ValueError(f"Expected list, got {type(queries)}")

        # 过滤空字符串，限制数量
        queries = [q.strip() for q in queries if isinstance(q, str) and q.strip()]
        return queries[:n]

    except Exception as e:
        tqdm.write(f"  [!] {ttp_id} 生成失败：{e}")
        return []


# ────────────────────────────────────────────────────────────────────────────
# 并发批量生成
# ────────────────────────────────────────────────────────────────────────────

def batch_generate(
    llm: ChatOpenAI,
    docs: list[dict],
    n_queries: int,
    batch_size: int,
    cache: dict,
) -> dict:
    """
    并发调用 LLM，返回 {ttp_id: [query1, query2, ...]} 的完整映射。
    已在 cache 中的条目直接跳过。
    """
    results = dict(cache)  # 从缓存起步
    todo = [d for d in docs if d["id"] not in results]

    print(f"[+] 需要生成：{len(todo)} 条（已缓存：{len(results)} 条）")

    for i in tqdm(range(0, len(todo), batch_size), desc="生成合成查询"):
        batch = todo[i: i + batch_size]

        # 构造 batch 输入
        messages_batch = [
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": build_user_prompt(
                    d["name"], d["description"], n_queries
                )},
            ]
            for d in batch
        ]

        try:
            responses = llm.batch(
                messages_batch,
                config={"max_concurrency": batch_size},
            )
            for doc, resp in zip(batch, responses):
                if isinstance(resp, Exception):
                    tqdm.write(f"  [!] {doc['id']} 批次内失败：{resp}")
                    results[doc["id"]] = []
                    continue

                raw = resp.content.strip()
                if raw.startswith("```"):
                    raw = raw.split("```")[1]
                    if raw.startswith("json"):
                        raw = raw[4:]
                raw = raw.strip()

                try:
                    queries = json.loads(raw)
                    queries = [q.strip() for q in queries
                               if isinstance(q, str) and q.strip()]
                    results[doc["id"]] = queries[:n_queries]
                except Exception as e:
                    tqdm.write(f"  [!] {doc['id']} 解析失败：{e}")
                    results[doc["id"]] = []

        except Exception as e:
            tqdm.write(f"  [!] 批次 {i//batch_size} 整体失败：{e}")
            for doc in batch:
                results[doc["id"]] = []

        # 每批完成后写缓存，支持断点续跑
        with open(D2Q_CACHE_PATH, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

    return results


# ────────────────────────────────────────────────────────────────────────────
# 建库：把合成查询存入向量库，原始文档存入 docstore
# ────────────────────────────────────────────────────────────────────────────

def build_index(
    docs: list[dict],
    query_map: dict,
    embedding_model: DashScopeEmbeddings,
):
    """
    将合成查询作为 child embedding 存入 ChromaDB。
    每个合成查询的 metadata 包含 parent_doc_id，
    命中后可从 docstore 取回完整 TTP 描述。
    """
    Path(D2Q_VECTORDB_DIR).mkdir(parents=True, exist_ok=True)
    Path(D2Q_DOCSTORE_DIR).mkdir(parents=True, exist_ok=True)

    vectordb = Chroma(
        collection_name="doc2query_chunks",
        persist_directory=D2Q_VECTORDB_DIR,
        embedding_function=embedding_model,
    )
    fs = LocalFileStore(D2Q_DOCSTORE_DIR)
    docstore = create_kv_docstore(fs)

    child_docs   = []   # 合成查询 Document（带 parent_id metadata）
    parent_docs  = {}   # {parent_id: Document}（原始 TTP）
    stats = {"total_queries": 0, "empty_ttps": 0}

    for doc in docs:
        ttp_id   = doc["id"]
        queries  = query_map.get(ttp_id, [])

        if not queries:
            stats["empty_ttps"] += 1
            # 没有合成查询时退化：直接用 description 分句作为 child
            fallback = doc["description"][:500]
            queries  = [fallback] if fallback else []

        # 为每个 TTP 生成稳定的 parent_id（用 TTP ID 的 hash）
        parent_id = hashlib.md5(ttp_id.encode()).hexdigest()

        # 存父文档
        parent_doc = Document(
            page_content=doc["description"],
            metadata={"id": ttp_id, "name": doc["name"]},
        )
        parent_docs[parent_id] = parent_doc

        # 为每条合成查询创建 child Document
        for idx, query in enumerate(queries):
            child_doc = Document(
                page_content=query,
                metadata={
                    "id":        ttp_id,    # 评估时用这个字段判断是否命中
                    "name":      doc["name"],
                    "doc_id":    parent_id, # ParentDocumentRetriever 标准字段
                },
            )
            child_docs.append(child_doc)
            stats["total_queries"] += 1

    # 写入 docstore（父文档）
    print(f"[+] 写入 {len(parent_docs)} 条父文档到 docstore...")
    docstore.mset(list(parent_docs.items()))

    # 写入 vectordb（合成查询 embedding）
    print(f"[+] 写入 {len(child_docs)} 条合成查询到向量库...")
    batch_size = 100
    for i in tqdm(range(0, len(child_docs), batch_size), desc="Embedding"):
        batch = child_docs[i: i + batch_size]
        vectordb.add_documents(batch)

    print(f"\n[+] 建库完成")
    print(f"    父文档数       : {len(parent_docs)}")
    print(f"    合成查询总数   : {stats['total_queries']}")
    print(f"    平均每 TTP     : {stats['total_queries']/len(docs):.1f} 条")
    print(f"    无合成查询 TTP : {stats['empty_ttps']} 条（已用 description 兜底）")
    print(f"    向量库路径     : {D2Q_VECTORDB_DIR}")
    print(f"    父文档路径     : {D2Q_DOCSTORE_DIR}")


# ────────────────────────────────────────────────────────────────────────────
# 入口
# ────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Doc2Query 建库")
    parser.add_argument("--n_queries",  type=int, default=5,
                        help="每条 TTP 生成的合成查询数量（默认 5）")
    parser.add_argument("--batch_size", type=int, default=20,
                        help="LLM 并发批大小（默认 20）")
    parser.add_argument("--resume",     action="store_true",
                        help="从缓存断点续跑，跳过已生成的条目")
    args = parser.parse_args()

    # 1. 加载 TTP 数据
    print(f"[+] 加载 TTP 数据：{DATASET2_PATH}")
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
    print(f"[+] 有效 TTP 数量：{len(docs)}")

    # 2. 初始化 LLM
    llm = ChatOpenAI(
        model="qwen-plus-latest",
        temperature=0.7,        # 提高多样性
        api_key=os.environ.get("DASHSCOPE_API_KEY"),
        base_url=os.environ.get("DASHSCOPE_API_URL"),
    )

    # 3. 加载缓存（断点续跑）
    cache = {}
    if args.resume and Path(D2Q_CACHE_PATH).exists():
        with open(D2Q_CACHE_PATH, encoding="utf-8") as f:
            cache = json.load(f)
        print(f"[+] 加载缓存：{len(cache)} 条已生成")

    # 4. 批量生成合成查询
    Path(D2Q_CACHE_PATH).parent.mkdir(parents=True, exist_ok=True)
    print(f"\n[+] 开始生成合成查询（n={args.n_queries}, batch={args.batch_size}）...")
    print(f"    预计 API 调用次数：{max(0, len(docs) - len(cache))}")
    print(f"    预计耗时：约 {max(0, len(docs)-len(cache)) / args.batch_size * 15 / 60:.0f} 分钟\n")

    query_map = batch_generate(llm, docs, args.n_queries, args.batch_size, cache)

    # 5. 建向量库
    print(f"\n[+] 初始化 Embedding 模型...")
    embedding_model = DashScopeEmbeddings(
        model=os.environ.get("EMBEDDING_MODEL")
    )
    build_index(docs, query_map, embedding_model)

    # 6. 打印示例
    sample_id = docs[0]["id"]
    print(f"\n[+] 示例（{sample_id} - {docs[0]['name']}）：")
    for i, q in enumerate(query_map.get(sample_id, [])[:3], 1):
        print(f"    [{i}] {q}")


if __name__ == "__main__":
    main()