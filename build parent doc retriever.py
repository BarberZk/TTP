"""
build_parent_doc_retriever.py
------------------------------
用 Parent Document Retriever 策略重建向量库。

核心思路：
  - 子 chunk（小）→ 存入 ChromaDB，用于 embedding 匹配（精准）
  - 父文档（完整）→ 存入 LocalFileStore，命中后返回给 LLM（上下文完整）

建库完成后，直接运行 evaluate_retrieval.py --mode parent 即可评估。

使用方式：
    python build_parent_doc_retriever.py
    python build_parent_doc_retriever.py --chunk_size 300 --chunk_overlap 50
"""

import os
import argparse
from pathlib import Path
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv(Path(".") / ".env")

from langchain_chroma import Chroma
from langchain_community.embeddings import DashScopeEmbeddings
from langchain.storage import LocalFileStore
from langchain_core.documents import Document
from langchain.storage import create_kv_docstore
from langchain_classic.retrievers import ParentDocumentRetriever
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ── 路径配置 ─────────────────────────────────────────────────────────────────
DATASET2_PATH         = "./data/input/MITRE-ATTACK_dataset_test.json"
PARENT_VECTORDB_DIR   = "./data/database/ttp_parent_chroma_db/"
PARENT_DOCSTORE_DIR   = "./data/database/ttp_parent_docstore/"


# ────────────────────────────────────────────────────────────────────────────
# 加载 MITRE ATT&CK 文档
# ────────────────────────────────────────────────────────────────────────────

def load_ttp_documents(path: str) -> list[Document]:
    df = pd.read_json(path, orient="records")
    required = ["ID", "name", "description"]
    assert all(c in df.columns for c in required), f"缺少必要列：{required}"

    docs, skipped = [], 0
    for _, row in df.iterrows():
        desc = row["description"]
        if pd.isna(desc) or not isinstance(desc, str) or not desc.strip():
            skipped += 1
            continue
        docs.append(Document(
            page_content=desc,
            metadata={"id": row["ID"], "name": row["name"]},
        ))

    print(f"[+] 加载 {len(docs)} 条 TTP 文档（跳过 {skipped} 条空描述）")
    return docs


# ────────────────────────────────────────────────────────────────────────────
# 建库
# ────────────────────────────────────────────────────────────────────────────

def build(chunk_size: int, chunk_overlap: int):
    # 1. 加载父文档
    parent_docs = load_ttp_documents(DATASET2_PATH)

    # 2. 子 chunk 切分器（用于 embedding）
    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", " "],  # 优先按段落切，其次按句子
    )

    # 3. embedding 模型（与 naive RAG 一致）
    embedding_model = DashScopeEmbeddings(
        model=os.environ.get("EMBEDDING_MODEL")
    )

    # 4. 子 chunk 向量库（新目录，不覆盖原始库）
    vectordb = Chroma(
        collection_name="child_chunks",
        persist_directory=PARENT_VECTORDB_DIR,
        embedding_function=embedding_model,
    )

    # 5. 父文档存储（LocalFileStore 持久化，key = doc_id）
    Path(PARENT_DOCSTORE_DIR).mkdir(parents=True, exist_ok=True)
    fs = LocalFileStore(PARENT_DOCSTORE_DIR)
    docstore = create_kv_docstore(fs)

    # 6. 组装 ParentDocumentRetriever
    retriever = ParentDocumentRetriever(
        vectorstore=vectordb,
        docstore=docstore,
        child_splitter=child_splitter,
        # parent_splitter=None 表示整条 TTP description 作为父文档
    )

    # 7. 写入（自动切 chunk + embed + 建父子映射）
    print(f"[+] 开始建库（chunk_size={chunk_size}, overlap={chunk_overlap}）...")
    print(f"    子 chunk 向量库 → {PARENT_VECTORDB_DIR}")
    print(f"    父文档存储     → {PARENT_DOCSTORE_DIR}")

    # ParentDocumentRetriever.add_documents 批量处理
    batch_size = 50
    for i in tqdm(range(0, len(parent_docs), batch_size), desc="建库进度"):
        batch = parent_docs[i: i + batch_size]
        retriever.add_documents(batch, ids=None)

    # 验证
    child_count = vectordb._collection.count()
    print(f"\n[+] 建库完成")
    print(f"    父文档数：{len(parent_docs)}")
    print(f"    子 chunk 数：{child_count}")
    print(f"    平均每文档切分：{child_count / len(parent_docs):.1f} 个 chunk")


# ────────────────────────────────────────────────────────────────────────────
# 入口
# ────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="构建 Parent Document Retriever 向量库")
    parser.add_argument("--chunk_size",    type=int, default=300,
                        help="子 chunk 字符数上限（默认 300）")
    parser.add_argument("--chunk_overlap", type=int, default=50,
                        help="相邻 chunk 重叠字符数（默认 50）")
    args = parser.parse_args()

    build(args.chunk_size, args.chunk_overlap)


if __name__ == "__main__":
    main()