"""
rag_tool.py
-----------
Hybrid 三路混合检索工具，供 LangGraph CTI Agent 调用。
检索策略：BM25（关键词精确匹配）+ Parent Document Retriever（语义检索）→ RRF 融合
Hit Rate@3 = 62.5%，MRR@3 = 0.5558
"""

import os
import re
import pickle
from pathlib import Path
from dotenv import load_dotenv

from langchain_core.tools import tool
from pydantic import BaseModel, Field
from langchain_chroma import Chroma
from langchain_community.embeddings import DashScopeEmbeddings
from langchain.storage import LocalFileStore, create_kv_docstore
from langchain_classic.retrievers import ParentDocumentRetriever
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# __file__ 是 agent/rag_tool.py，.parent 是 agent/，.parent.parent 是项目根目录
_ROOT = Path(__file__).parent.parent

env_path = _ROOT / '.env'
load_dotenv(dotenv_path=env_path)

# ── 路径配置（基于项目根目录的绝对路径，无论从哪里运行都正确）────────────────
PARENT_VECTORDB_DIR = str(_ROOT / "data/database/ttp_parent_chroma_db/")
PARENT_DOCSTORE_DIR = str(_ROOT / "data/database/ttp_parent_docstore/")
BM25_INDEX_PATH     = str(_ROOT / "data/database/ttp_bm25_index.pkl")

RRF_K = 60   # RRF 平滑常数，防止头部排名差异过大地主导结果
TOP_K = 3    # 最终返回父文档数


# ── 初始化检索组件（模块级单例，只初始化一次）──────────────────────────────────
def _tokenize(text: str) -> list:
    """BM25 分词：小写 + 按非字母数字切分，保留连字符词"""
    return re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)*", text.lower())


def _init_retrievers():
    embedding_model = DashScopeEmbeddings(
        model=os.environ.get("EMBEDDING_MODEL", "text-embedding-v3")
    )

    # Parent Document Retriever（语义路）
    vectordb = Chroma(
        collection_name="child_chunks",
        persist_directory=PARENT_VECTORDB_DIR,
        embedding_function=embedding_model,
    )
    docstore = create_kv_docstore(LocalFileStore(PARENT_DOCSTORE_DIR))
    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300, chunk_overlap=50,
        separators=["\n\n", "\n", ".", " "],
    )
    parent_retriever = ParentDocumentRetriever(
        vectorstore=vectordb,
        docstore=docstore,
        child_splitter=child_splitter,
        search_kwargs={"k": TOP_K * 4},
    )

    # BM25（关键词路）
    with open(BM25_INDEX_PATH, "rb") as f:
        bm25_data = pickle.load(f)

    return parent_retriever, bm25_data


try:
    _parent_retriever, _bm25_data = _init_retrievers()
    print("✅ Hybrid 检索器初始化成功（BM25 + Parent Doc + RRF）")
    _retriever_ready = True
except Exception as e:
    print(f"❌ Hybrid 检索器初始化失败：{e}")
    _parent_retriever, _bm25_data = None, None
    _retriever_ready = False


def _hybrid_search(query: str) -> list:
    """
    执行 BM25 + 向量 RRF 融合检索，返回 top-k 父文档。

    RRF 公式：score(d) = 1/(rank_BM25 + 60) + 1/(rank_vector + 60)
    两路都认可的文档得分叠加，只有一路命中的文档得分降低。
    """
    pool_size = TOP_K * 4

    # BM25 路
    bm25      = _bm25_data["bm25"]
    bm25_docs = _bm25_data["docs"]
    scores    = bm25.get_scores(_tokenize(query))
    top_idx   = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:pool_size]
    bm25_ranks = {bm25_docs[i]["id"]: rank + 1 for rank, i in enumerate(top_idx)}

    # 向量路
    vector_docs  = _parent_retriever.invoke(query)
    vector_ranks = {doc.metadata.get("id", ""): rank + 1
                    for rank, doc in enumerate(vector_docs)}

    # RRF 融合
    all_ids = set(bm25_ranks) | set(vector_ranks)
    rrf_scores = {}
    for tid in all_ids:
        score = 0.0
        if tid in bm25_ranks:
            score += 1.0 / (bm25_ranks[tid] + RRF_K)
        if tid in vector_ranks:
            score += 1.0 / (vector_ranks[tid] + RRF_K)
        rrf_scores[tid] = score

    top_ids    = sorted(rrf_scores, key=lambda x: rrf_scores[x], reverse=True)[:TOP_K]
    id_to_meta = {d["id"]: d for d in bm25_docs}

    results = []
    for tid in top_ids:
        if tid in id_to_meta:
            d = id_to_meta[tid]
            results.append(Document(
                page_content=d["description"],
                metadata={"id": d["id"], "name": d["name"]},
            ))
    return results


# ── LangChain Tool 封装 ────────────────────────────────────────────────────────

class SearchAttackInput(BaseModel):
    query: str = Field(
        description="需要查询的具体攻击行为、技术名称或战术关键词，"
                    "例如 'Registry persistence' 或 'VBA macro execution'"
    )


@tool("search_mitre_attack_knowledge_base", args_schema=SearchAttackInput)
def search_mitre_attack_tool(query: str) -> str:
    """
    当你不确定某段文本属于哪个 MITRE ATT&CK 战术(Tactic)或技术(Technique)时，
    或者你需要获取官方定义来辅助判断时，请使用此工具。
    输入应该是一段描述攻击行为的英文或中文短语。
    本工具使用 BM25 + 向量语义 + RRF 融合，Hit Rate@3 = 62.5%。
    """
    if not _retriever_ready:
        return "Error: 知识库未初始化，请先运行 build_parent_doc_retriever.py 和 build_bm25_index.py。"

    try:
        docs = _hybrid_search(query)
        if not docs:
            return "知识库中未找到高度相关的 TTP 定义，请尝试更换检索词。"

        results = []
        for doc in docs:
            ttp_id   = doc.metadata.get("id",   "Unknown ID")
            ttp_name = doc.metadata.get("name", "Unknown Name")
            desc     = doc.page_content[:500]
            results.append(f"TTP ID: {ttp_id}\n名称: {ttp_name}\n官方描述: {desc}...")

        return "\n---\n".join(results)

    except Exception as e:
        return f"检索过程中发生错误: {str(e)}"


if __name__ == "__main__":
    print("--- 正在测试 Hybrid Tool 调用 ---")
    result = search_mitre_attack_tool.invoke({"query": "TrickBot macro download malware"})
    print(result)