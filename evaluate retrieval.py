"""
evaluate_retrieval.py
---------------------
评估 ChromaDB 检索器的召回质量。
指标：Hit Rate@k 和 MRR@k

使用方式：
    python evaluate_retrieval.py --mode naive
    python evaluate_retrieval.py --mode parent
    python evaluate_retrieval.py --mode hybrid     # BM25 + 向量 RRF 融合
    python evaluate_retrieval.py --mode doc2query  # Doc2Query 合成查询索引
    python evaluate_retrieval.py --mode compare    # 对比所有已保存结果
    python evaluate_retrieval.py --mode naive --k 5 --sample 500
    python evaluate_retrieval.py --mode naive --sample 0  # 全量
"""

import os
import json
import pickle
import re
import argparse
import random
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict
from dotenv import load_dotenv

load_dotenv(Path(".") / ".env")

from langchain_chroma import Chroma
from langchain_community.embeddings import DashScopeEmbeddings
from langchain.storage import LocalFileStore, create_kv_docstore
from langchain_classic.retrievers import ParentDocumentRetriever
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ── 路径配置 ─────────────────────────────────────────────────────────────────
DATASET1_PATH       = "./data/input/sft_dataset_4000.json"
VECTORDB_NAIVE_DIR  = "./data/database/ttp_chroma_db_qwen_compat/"
PARENT_VECTORDB_DIR = "./data/database/ttp_parent_chroma_db/"
PARENT_DOCSTORE_DIR = "./data/database/ttp_parent_docstore/"
BM25_INDEX_PATH     = "./data/database/ttp_bm25_index.pkl"
EVAL_NAIVE_PATH     = "./data/output/eval_naive.json"
EVAL_PARENT_PATH    = "./data/output/eval_parent.json"
EVAL_HYBRID_PATH    = "./data/output/eval_hybrid.json"
EVAL_D2Q_PATH        = "./data/output/eval_doc2query.json"
EVAL_HYBRID_D2Q_PATH = "./data/output/eval_hybrid_doc2query.json"
D2Q_VECTORDB_DIR    = "./data/database/ttp_doc2query_chroma_db/"
D2Q_DOCSTORE_DIR    = "./data/database/ttp_doc2query_docstore/"


# ────────────────────────────────────────────────────────────────────────────
# 工具函数
# ────────────────────────────────────────────────────────────────────────────

def extract_ttp_id(output_str: str) -> str:
    return output_str.split(":")[0].strip()


def is_hit(correct_id: str, retrieved_id: str) -> bool:
    if correct_id == retrieved_id:
        return True
    if retrieved_id.startswith(correct_id + "."):
        return True
    return False


def find_hit_rank(correct_id: str, retrieved_ids: list) -> int | None:
    for rank, rid in enumerate(retrieved_ids, start=1):
        if is_hit(correct_id, rid):
            return rank
    return None


def tokenize(text: str) -> list:
    text = text.lower()
    return re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)*", text)


def load_dataset(path: str, sample_size: int) -> list:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    valid = [
        item for item in data
        if isinstance(item, dict)
        and item.get("input", "").strip()
        and item.get("output", "").strip()
    ]
    if sample_size and sample_size < len(valid):
        random.seed(42)
        valid = random.sample(valid, sample_size)
    return valid


# ────────────────────────────────────────────────────────────────────────────
# 检索器构建
# ────────────────────────────────────────────────────────────────────────────

def get_embedding_model():
    return DashScopeEmbeddings(model=os.environ.get("EMBEDDING_MODEL"))


def build_naive_retriever(k: int):
    vectordb = Chroma(
        persist_directory=VECTORDB_NAIVE_DIR,
        embedding_function=get_embedding_model(),
    )
    retriever = vectordb.as_retriever(search_kwargs={"k": k})
    print(f"[+] Naive 检索器就绪，向量库 {vectordb._collection.count()} 条，top-{k}")
    return retriever


def build_parent_retriever(k: int):
    for path in [PARENT_VECTORDB_DIR, PARENT_DOCSTORE_DIR]:
        assert Path(path).exists(), f"找不到 {path}，请先运行 build_parent_doc_retriever.py"

    vectordb = Chroma(
        collection_name="child_chunks",
        persist_directory=PARENT_VECTORDB_DIR,
        embedding_function=get_embedding_model(),
    )
    docstore = create_kv_docstore(LocalFileStore(PARENT_DOCSTORE_DIR))
    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300, chunk_overlap=50,
        separators=["\n\n", "\n", ".", " "],
    )
    child_k = k * 4
    retriever = ParentDocumentRetriever(
        vectorstore=vectordb,
        docstore=docstore,
        child_splitter=child_splitter,
        search_kwargs={"k": child_k},
    )
    print(f"[+] Parent 检索器就绪，子 chunk {vectordb._collection.count()} 条")
    print(f"    子 chunk top-{child_k} -> 去重后目标父文档 top-{k}")
    return retriever


def build_doc2query_retriever(k: int):
    """加载 Doc2Query 检索器：合成查询作为 child，原始 TTP 作为 parent。"""
    import hashlib
    for path in [D2Q_VECTORDB_DIR, D2Q_DOCSTORE_DIR]:
        assert Path(path).exists(), (
            f"找不到 {path}\n请先运行：python build_doc2query_index.py"
        )
    vectordb = Chroma(
        collection_name="doc2query_chunks",
        persist_directory=D2Q_VECTORDB_DIR,
        embedding_function=get_embedding_model(),
    )
    docstore = create_kv_docstore(LocalFileStore(D2Q_DOCSTORE_DIR))
    child_splitter = RecursiveCharacterTextSplitter(chunk_size=9999)
    child_k = k * 4
    retriever = ParentDocumentRetriever(
        vectorstore=vectordb,
        docstore=docstore,
        child_splitter=child_splitter,
        search_kwargs={"k": child_k},
    )
    print(f"[+] Doc2Query 检索器就绪，合成查询 {vectordb._collection.count()} 条")
    print(f"    child top-{child_k} -> 去重后目标父文档 top-{k}")
    return retriever


def build_hybrid_retriever(k: int):
    """
    混合检索器：BM25（关键词）+ 向量（语义） → RRF 融合排名。

    RRF 公式：score(d) = Σ 1 / (rank_i(d) + 60)
    60 是平滑常数，防止头部排名差异过大地主导结果。
    """
    assert Path(BM25_INDEX_PATH).exists(), (
        f"找不到 BM25 索引：{BM25_INDEX_PATH}\n"
        f"请先运行：python build_bm25_index.py"
    )
    assert Path(PARENT_VECTORDB_DIR).exists(), (
        f"找不到 Parent 向量库：{PARENT_VECTORDB_DIR}\n"
        f"请先运行：python build_parent_doc_retriever.py"
    )

    with open(BM25_INDEX_PATH, "rb") as f:
        bm25_data = pickle.load(f)

    vectordb = Chroma(
        collection_name="child_chunks",
        persist_directory=PARENT_VECTORDB_DIR,
        embedding_function=get_embedding_model(),
    )
    docstore = create_kv_docstore(LocalFileStore(PARENT_DOCSTORE_DIR))
    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300, chunk_overlap=50,
        separators=["\n\n", "\n", ".", " "],
    )
    # 向量侧也用 parent retriever，返回完整父文档
    vector_retriever = ParentDocumentRetriever(
        vectorstore=vectordb,
        docstore=docstore,
        child_splitter=child_splitter,
        search_kwargs={"k": k * 4},
    )


    print(f"[+] Hybrid 检索器就绪（BM25 + 向量 RRF），top-{k}")
    return {"bm25": bm25_data, "vector": vector_retriever, "k": k}


def build_hybrid_doc2query_retriever(k: int):
    """
    三路融合检索器：BM25 + Parent向量 + Doc2Query向量 → RRF。
    Doc2Query 提供语义桥接，BM25 提供关键词精准，Parent向量提供上下文语义。
    """
    for path, name in [
        (BM25_INDEX_PATH,    "BM25索引，运行 build_bm25_index.py"),
        (PARENT_VECTORDB_DIR,"Parent向量库，运行 build_parent_doc_retriever.py"),
        (D2Q_VECTORDB_DIR,   "Doc2Query向量库，运行 build_doc2query_index.py"),
    ]:
        assert Path(path).exists(), f"找不到 {path}\n请先运行：{name}"

    with open(BM25_INDEX_PATH, "rb") as f:
        bm25_data = pickle.load(f)

    emb = get_embedding_model()

    # 向量路1：parent chunk 语义检索
    parent_vectordb = Chroma(
        collection_name="child_chunks",
        persist_directory=PARENT_VECTORDB_DIR,
        embedding_function=emb,
    )
    parent_retriever = ParentDocumentRetriever(
        vectorstore=parent_vectordb,
        docstore=create_kv_docstore(LocalFileStore(PARENT_DOCSTORE_DIR)),
        child_splitter=RecursiveCharacterTextSplitter(
            chunk_size=300, chunk_overlap=50,
            separators=["\n\n", "\n", ".", " "],
        ),
        search_kwargs={"k": k * 4},
    )

    # 向量路2：doc2query 合成查询检索
    d2q_vectordb = Chroma(
        collection_name="doc2query_chunks",
        persist_directory=D2Q_VECTORDB_DIR,
        embedding_function=emb,
    )
    d2q_retriever = ParentDocumentRetriever(
        vectorstore=d2q_vectordb,
        docstore=create_kv_docstore(LocalFileStore(D2Q_DOCSTORE_DIR)),
        child_splitter=RecursiveCharacterTextSplitter(chunk_size=9999),
        search_kwargs={"k": k * 4},
    )

    print(f"[+] Hybrid-Doc2Query 检索器就绪（BM25 + Parent向量 + D2Q向量 RRF），top-{k}")
    return {
        "bm25":          bm25_data,
        "parent_ret":    parent_retriever,
        "d2q_ret":       d2q_retriever,
        "k":             k,
    }


def hybrid_doc2query_invoke(retriever_bundle: dict, query: str) -> list:
    """
    三路 RRF 融合：BM25 + Parent向量 + Doc2Query向量。
    """
    from langchain_core.documents import Document

    bm25      = retriever_bundle["bm25"]["bm25"]
    bm25_docs = retriever_bundle["bm25"]["docs"]
    parent_ret = retriever_bundle["parent_ret"]
    d2q_ret    = retriever_bundle["d2q_ret"]
    k: int     = retriever_bundle["k"]
    rrf_k: int = 60
    pool_size  = k * 4

    # BM25
    bm25_scores  = bm25.get_scores(tokenize(query))
    bm25_top_idx = sorted(range(len(bm25_scores)),
                          key=lambda i: bm25_scores[i], reverse=True)[:pool_size]
    bm25_ranks   = {bm25_docs[i]["id"]: rank + 1
                    for rank, i in enumerate(bm25_top_idx)}

    # Parent 向量
    parent_docs  = parent_ret.invoke(query)
    parent_ranks = {doc.metadata.get("id", ""): rank + 1
                    for rank, doc in enumerate(parent_docs)}

    # Doc2Query 向量
    d2q_docs   = d2q_ret.invoke(query)
    d2q_ranks  = {doc.metadata.get("id", ""): rank + 1
                  for rank, doc in enumerate(d2q_docs)}

    # RRF 三路融合
    all_ids = set(bm25_ranks) | set(parent_ranks) | set(d2q_ranks)
    rrf_scores = {}
    for tid in all_ids:
        score = 0.0
        if tid in bm25_ranks:
            score += 1.0 / (bm25_ranks[tid] + rrf_k)
        if tid in parent_ranks:
            score += 1.0 / (parent_ranks[tid] + rrf_k)
        if tid in d2q_ranks:
            score += 1.0 / (d2q_ranks[tid] + rrf_k)
        rrf_scores[tid] = score

    top_ids   = sorted(rrf_scores, key=lambda x: rrf_scores[x], reverse=True)[:k]
    id_to_doc = {d["id"]: d for d in bm25_docs}
    result    = []
    for tid in top_ids:
        if tid in id_to_doc:
            d = id_to_doc[tid]
            result.append(Document(
                page_content=d["description"],
                metadata={"id": d["id"], "name": d["name"]},
            ))
    return result


def hybrid_invoke(retriever_bundle: dict, query: str) -> list:
    """
    执行混合检索，返回 RRF 融合后的父文档列表（已截断到 top-k）。
    """
    bm25: object    = retriever_bundle["bm25"]["bm25"]
    docs: list      = retriever_bundle["bm25"]["docs"]
    vector_ret      = retriever_bundle["vector"]
    k: int          = retriever_bundle["k"]
    rrf_k: int      = 60   # RRF 平滑常数

    # ── BM25 侧：返回 top pool_size 个 TTP 父文档 ─────────────────────────
    pool_size = k * 4
    query_tokens = tokenize(query)
    bm25_scores  = bm25.get_scores(query_tokens)
    bm25_top_idx = sorted(range(len(bm25_scores)),
                          key=lambda i: bm25_scores[i], reverse=True)[:pool_size]
    # {ttp_id: rank (1-indexed)}
    bm25_ranks = {docs[i]["id"]: rank + 1 for rank, i in enumerate(bm25_top_idx)}

    # ── 向量侧：Parent retriever 返回父文档 ──────────────────────────────
    vector_docs = vector_ret.invoke(query)
    vector_ranks = {doc.metadata.get("id", ""): rank + 1
                    for rank, doc in enumerate(vector_docs)}

    # ── RRF 融合 ──────────────────────────────────────────────────────────
    all_ids = set(bm25_ranks) | set(vector_ranks)
    rrf_scores = {}
    for tid in all_ids:
        score = 0.0
        if tid in bm25_ranks:
            score += 1.0 / (bm25_ranks[tid] + rrf_k)
        if tid in vector_ranks:
            score += 1.0 / (vector_ranks[tid] + rrf_k)
        rrf_scores[tid] = score

    # 按 RRF 分数排序，取 top-k
    top_ids = sorted(rrf_scores, key=lambda x: rrf_scores[x], reverse=True)[:k]

    # 构造返回格式与其他检索器一致（带 metadata 的 Document 对象）
    from langchain_core.documents import Document
    id_to_doc = {d["id"]: d for d in docs}
    result = []
    for tid in top_ids:
        if tid in id_to_doc:
            d = id_to_doc[tid]
            result.append(Document(
                page_content=d["description"],
                metadata={"id": d["id"], "name": d["name"]},
            ))
    return result


# ────────────────────────────────────────────────────────────────────────────
# 核心评估
# ────────────────────────────────────────────────────────────────────────────

def evaluate(retriever, dataset: list, k: int, mode: str) -> dict:
    hits = 0
    reciprocal_ranks = []
    rank_distribution = defaultdict(int)
    miss_examples = []

    for item in tqdm(dataset, desc=f"Evaluating [{mode}]"):
        query      = item["input"]
        correct_id = extract_ttp_id(item["output"])

        if mode == "hybrid":
            retrieved_docs = hybrid_invoke(retriever, query)
        elif mode == "hybrid_doc2query":
            retrieved_docs = hybrid_doc2query_invoke(retriever, query)
        else:
            retrieved_docs = retriever.invoke(query)

        retrieved_ids = [
            doc.metadata.get("id", "") for doc in retrieved_docs
        ][:k]  # 严格截断到 top-k，保证评估口径一致

        rank = find_hit_rank(correct_id, retrieved_ids)

        if rank is not None:
            hits += 1
            reciprocal_ranks.append(1.0 / rank)
            rank_distribution[rank] += 1
        else:
            reciprocal_ranks.append(0.0)
            if len(miss_examples) < 5:
                miss_examples.append({
                    "input":      query[:80] + "...",
                    "correct_id": correct_id,
                    "retrieved":  retrieved_ids,
                })

    total = len(dataset)
    return {
        "hit_rate":          hits / total,
        "mrr":               sum(reciprocal_ranks) / total,
        "total":             total,
        "hits":              hits,
        "miss_examples":     miss_examples,
        "rank_distribution": dict(rank_distribution),
    }


# ────────────────────────────────────────────────────────────────────────────
# 报告打印
# ────────────────────────────────────────────────────────────────────────────

def print_report(result: dict, k: int, label: str = ""):
    sep = "-" * 50
    title = f"  检索评估报告  (top-k = {k})"
    if label:
        title += f"  [{label}]"
    print(f"\n{sep}")
    print(title)
    print(sep)
    print(f"  样本总量      : {result['total']}")
    print(f"  命中数        : {result['hits']}")
    print(f"  Hit Rate@{k}   : {result['hit_rate']:.4f}  ({result['hit_rate']*100:.1f}%)")
    print(f"  MRR@{k}        : {result['mrr']:.4f}")
    print(f"\n  命中排名分布：")
    for rank in sorted(result["rank_distribution"]):
        count = result["rank_distribution"][rank]
        bar   = "#" * int(count / result["total"] * 40)
        print(f"    第 {rank} 位: {count:4d} 条  {bar}")
    if result["miss_examples"]:
        print(f"\n  未命中示例（前 {len(result['miss_examples'])} 条）：")
        for i, ex in enumerate(result["miss_examples"], 1):
            print(f"\n  [{i}] Input   : {ex['input']}")
            print(f"      正确 ID  : {ex['correct_id']}")
            print(f"      召回 IDs : {ex['retrieved']}")
    print(f"\n{sep}\n")


def print_comparison(results: dict, k: int):
    sep = "-" * 50
    labels  = list(results.keys())
    sample  = list(results.values())[0]["total"]

    print(f"\n{sep}")
    print(f"  对比报告  (top-k = {k}, 样本 = {sample} 条)")
    print(sep)

    # 表头
    col_w = 13
    header = f"  {'指标':<14}" + "".join(f"{l.upper():>{col_w}}" for l in labels)
    print(header)
    print(f"  {'-'*14}" + "-" * col_w * len(labels))

    # Hit Rate 行
    row = f"  {'Hit Rate@'+str(k):<14}"
    for l in labels:
        row += f"{results[l]['hit_rate']*100:>{col_w-1}.1f}%"
    print(row)

    # MRR 行
    row = f"  {'MRR@'+str(k):<14}"
    for l in labels:
        row += f"{results[l]['mrr']:>{col_w}.4f}"
    print(row)

    # 命中数行
    row = f"  {'命中数':<14}"
    for l in labels:
        row += f"{results[l]['hits']:>{col_w}}"
    print(row)

    print(f"\n{sep}\n")


# ────────────────────────────────────────────────────────────────────────────
# 入口
# ────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode",   choices=["naive", "parent", "hybrid", "doc2query", "hybrid_doc2query", "compare"],
                        default="naive")
    parser.add_argument("--k",      type=int, default=3)
    parser.add_argument("--sample", type=int, default=200)
    args = parser.parse_args()

    mode_to_path = {
        "naive":          EVAL_NAIVE_PATH,
        "parent":         EVAL_PARENT_PATH,
        "hybrid":         EVAL_HYBRID_PATH,
        "doc2query":      EVAL_D2Q_PATH,
        "hybrid_doc2query": EVAL_HYBRID_D2Q_PATH,
    }

    # compare 模式
    if args.mode == "compare":
        loaded = {}
        for mode, path in mode_to_path.items():
            if Path(path).exists():
                with open(path) as f:
                    loaded[mode] = json.load(f)
                print_report(loaded[mode], args.k, label=mode)
            else:
                print(f"[!] 跳过 {mode}（找不到 {path}）")
        if loaded:
            print_comparison(loaded, args.k)
        return

    # 单模式评估
    print(f"[+] 加载数据集：{DATASET1_PATH}")
    dataset = load_dataset(DATASET1_PATH, args.sample)
    print(f"[+] 评估样本数：{len(dataset)}，模式：{args.mode}")

    if args.mode == "naive":
        retriever = build_naive_retriever(args.k)
    elif args.mode == "parent":
        retriever = build_parent_retriever(args.k)
    elif args.mode == "doc2query":
        retriever = build_doc2query_retriever(args.k)
    elif args.mode == "hybrid_doc2query":
        retriever = build_hybrid_doc2query_retriever(args.k)
    else:
        retriever = build_hybrid_retriever(args.k)

    result = evaluate(retriever, dataset, args.k, args.mode)
    print_report(result, args.k, label=args.mode)

    out_path = mode_to_path[args.mode]
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"[+] 结果已保存到 {out_path}")


if __name__ == "__main__":
    main()