"""
main.py
-------
完整 TTP 提取流水线编排。

目录结构：
    project_root/
    ├── agent/          ← 本文件所在目录
    │   ├── main.py
    │   ├── agent_core.py
    │   └── rag_tool.py
    └── pdf/            ← 威胁情报 PDF 存放目录
        ├── report1.pdf
        └── report2.pdf

阶段 1：初始化 LangGraph Agent + 本地 SFT 模型
阶段 2：pdfplumber 解析 PDF → 三级语义切分 → 带元数据 chunk 列表
阶段 3：LangGraph Agent 流式阅读，携带结构化记忆，提取 TTP 候选句子
阶段 4：本地 SFT 模型精准映射 ATT&CK ID + CoT 推理
阶段 5：输出结构化 JSON 报告
"""

import json
import time
import re
from pathlib import Path
from collections import Counter
from tqdm import tqdm

import pdfplumber
from agent_core import CTILangGraphAgent
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import re as _re


def _parse_model_output(text: str) -> str:
    """
    从本地 SFT 模型的原始输出中提取干净的 ATT&CK ID + 技术名称。

    逻辑与 evaluation.ipynb 中的 extract_final_answer_id 保持一致：
    1. 按 [Final Answer]: 切分，取最后一段（处理模型重复输出多个 Final Answer 的情况）
    2. 从该段提取 Txxxx 或 Txxxx.xxx 格式的 ID，同时保留后面的技术名称
    3. 截断规则：遇到逗号/句号/换行/括号 + 解释文字时停止，只保留 "Txxxx: 技术名称"
    4. 如果找不到 Final Answer 标记，fallback 到全文搜索第一个 ATT&CK ID
    """
    if not text:
        return "Unknown TTP"

    # 按 [Final Answer]: 切分，取最后一段
    parts = _re.split(r'\[Final Answer\]\s*:', text, flags=_re.IGNORECASE)
    candidate = parts[-1].strip() if len(parts) > 1 else text.strip()

    # 在候选文本里匹配 Txxxx(.xxx): 技术名称
    # 匹配规则：ID + 可选冒号 + 可选技术名称（到逗号/句号/换行/括号为止）
    match = _re.search(
        r'(T\d{4}(?:\.\d{3})?)'           # ATT&CK ID
        r'(?:\s*:\s*'                        # 可选冒号
        r'([A-Za-z][A-Za-z0-9 /\-]{1,60}))?'  # 可选技术名称（最多 60 字符）
        r'(?=[,\.\n\(]|\s{2,}|$)',         # 在这些字符前截断
        candidate
    )
    if match:
        tid  = match.group(1)
        name = match.group(2)
        if name:
            return f"{tid}: {name.strip()}"
        return tid

    # fallback：全文搜索第一个 ATT&CK ID
    match = _re.search(r'T\d{4}(?:\.\d{3})?', text)
    if match:
        return match.group(0)

    return "Unknown TTP"



def _split_extracted(extracted: str) -> list[str]:
    """
    将 agent 输出的 extracted_sentence 拆分为干净的单句列表。

    agent_core 对多句话返回 JSON 数组字符串，单句直接返回字符串。
    本函数统一处理两种情况，返回 list[str]，每条只含一个 TTP 句子。
    """
    if not extracted:
        return []
    try:
        parsed = json.loads(extracted)
        if isinstance(parsed, list):
            return [s.strip() for s in parsed if s and len(s.strip()) > 10]
    except (json.JSONDecodeError, ValueError):
        pass
    # 单句字符串直接返回
    return [extracted.strip()] if extracted.strip() else []



# pdf 文件夹与 agent 文件夹同级
PDF_DIR     = Path(__file__).parent.parent / "pdf"
TRACE_DIR   = Path(__file__).parent / "data/output"


def _print_tool_log(tool_log_entry: dict):
    """
    实时打印单个 chunk 的工具调用情况。
    tool_called=False 时打印一行简短提示，True 时打印每次调用的查询词和召回结果。
    """
    if not tool_log_entry:
        return
    if not tool_log_entry.get("tool_called", False):
        print("    [RAG] 未调用工具（语义明确，直接推理）")
        return

    count = tool_log_entry.get("tool_call_count", 0)
    print(f"    [RAG] 调用工具 {count} 次：")
    for call in tool_log_entry.get("tool_calls", []):
        query = call.get("query", "")
        docs  = call.get("recalled_docs", [])
        doc_str = "  |  ".join(
            f"{d['ttp_id']}: {d['name']}" for d in docs
        ) if docs else "（无召回结果）"
        print(f'         ├─ 查询："{query}"')
        print(f"         └─ 召回：{doc_str}")


def _save_trace(label: str, trace_records: list):
    """
    把整篇文档所有 chunk 的工具调用记录写入 agent_trace_xxx.json。
    """
    TRACE_DIR.mkdir(parents=True, exist_ok=True)
    safe_name = re.sub(r"[^\w]", "_", label.replace(".pdf", ""))
    trace_path = TRACE_DIR / f"agent_trace_{safe_name}.json"
    with open(trace_path, "w", encoding="utf-8") as f:
        json.dump(trace_records, f, ensure_ascii=False, indent=4)
    print(f"[Trace] Agent 调用日志已保存至：{trace_path}")
MEMORY_FILE = Path(__file__).parent / "data/output/agent_memory.json"

# ── 持久化记忆工具函数 ────────────────────────────────────────────────────────

EMPTY_MEMORY = {
    "threat_actor":    "",
    "malware_names":   [],
    "targets":         [],
    "techniques_seen": []
}


def load_memory() -> str:
    """
    从文件加载历史记忆。
    不存在时返回空记忆，支持跨文档、跨次运行的 APT 组织画像累积。
    """
    if MEMORY_FILE.exists():
        try:
            raw = MEMORY_FILE.read_text(encoding="utf-8")
            json.loads(raw)   # 验证 JSON 合法性
            print(f"[Memory] 加载历史记忆：{MEMORY_FILE.name}")
            return raw
        except Exception:
            print("[Memory] 历史记忆文件损坏，使用空记忆重新开始")
    else:
        print("[Memory] 未发现历史记忆，使用空记忆初始化")
    return json.dumps(EMPTY_MEMORY, ensure_ascii=False)


def save_memory(summary: str):
    """
    处理完一份文档后，把最终记忆持久化到文件。
    JSON 损坏时跳过写入，保留上一次的有效记忆。
    """
    try:
        parsed = json.loads(summary)
        MEMORY_FILE.parent.mkdir(parents=True, exist_ok=True)
        MEMORY_FILE.write_text(
            json.dumps(parsed, ensure_ascii=False, indent=2),
            encoding="utf-8"
        )
        print(f"[Memory] 记忆已持久化 → {MEMORY_FILE.name}")
    except Exception as e:
        print(f"[Memory] 记忆写入失败（{e}），保留上次记忆不变")


def show_memory_diff(old_summary: str, new_summary: str):
    """打印本次运行新发现的实体，让用户直观看到记忆增量。"""
    try:
        old = json.loads(old_summary)
        new = json.loads(new_summary)
        new_techniques = [t for t in new.get("techniques_seen", [])
                          if t not in old.get("techniques_seen", [])]
        new_malware    = [m for m in new.get("malware_names", [])
                          if m not in old.get("malware_names", [])]
        if new_techniques or new_malware:
            print(f"\n[Memory] 本次新增内容：")
            if new_techniques:
                print(f"  techniques_seen +{len(new_techniques)}: {new_techniques}")
            if new_malware:
                print(f"  malware_names   +{len(new_malware)}: {new_malware}")
    except Exception:
        pass


# ══════════════════════════════════════════════════════════════════════
# 1. PDF 文档解析器
# ══════════════════════════════════════════════════════════════════════

# 需要整体跳过的章节关键词（不区分大小写，前缀匹配）
SKIP_SECTIONS = [
    "references", "bibliography", "about the author", "about the company",
    "about [", "legal disclaimer", "copyright", "table of contents",
    "indicators of compromise", "ioc", "appendix",
]

# 目标 chunk 词数范围
CHUNK_MIN_WORDS = 200
CHUNK_MAX_WORDS = 600
OVERLAP_WORDS   = 100   # 相邻 chunk 的 overlap 词数


class DocumentChunker:
    """
    PDF 三级语义切分器。

    切分优先级：
        一级 - 章节边界（字体大小 >= 正文 x1.2 且独占一行）
        二级 - 段落边界（连续两个换行符或行间距明显偏大）
        三级 - 句子边界（句号 + 空格 + 大写，仅在段落仍然过长时启用）

    目标 chunk 大小：300～600 词
    Overlap：固定 100 词，取自上一 chunk 末尾，拼接到下一 chunk 开头

    每个 chunk 携带 5 字段元数据：
        chunk_id    : 全局顺序编号（从 0 开始）
        section     : 所属章节标题
        page_range  : 来源页码（如 "3-4"）
        word_count  : 当前 chunk 词数
        is_first    : 是否是文档的第一个 chunk
    """

    def __init__(self, pdf_path: str | None = None):
        self.pdf_path = Path(pdf_path) if pdf_path else None

    # ── 公开接口 ──────────────────────────────────────────────────────

    def get_chunks(self) -> list[dict]:
        """返回 [{"text": str, "metadata": dict}, ...] 列表。"""
        if self.pdf_path and self.pdf_path.exists():
            return self._parse_real_pdf(self.pdf_path)
        return self._mock_chunks()

    # ── 真实 PDF 解析 ─────────────────────────────────────────────────

    def _parse_real_pdf(self, pdf_path: Path) -> list[dict]:
        """主解析入口：提取 → 清洗 → 切分 → 附加元数据。"""
        print(f"[Parser] 正在解析 PDF：{pdf_path.name}")

        with pdfplumber.open(str(pdf_path)) as pdf:
            total_pages = len(pdf.pages)

            # ── Step 1：提取每页文本块（保留 bbox 坐标用于过滤）
            raw_pages = self._extract_pages(pdf)

        # ── Step 2：识别并过滤页眉页脚
        cleaned_pages = self._remove_headers_footers(raw_pages)

        # ── Step 3：识别正文字体大小基准值
        body_font_size = self._detect_body_font_size(cleaned_pages)
        print(f"[Parser] 正文基准字体大小：{body_font_size:.1f}pt")

        # ── Step 4：合并成带章节标注的段落序列
        sections = self._build_sections(cleaned_pages, body_font_size)
        print(f"[Parser] 识别到 {len(sections)} 个章节")

        # ── Step 5：跳过无关章节（References / IOC 等）
        sections = self._filter_sections(sections)
        print(f"[Parser] 过滤后保留 {len(sections)} 个章节")

        # ── Step 6：三级切分 + Overlap
        chunks = self._split_and_overlap(sections)
        print(f"[Parser] 共生成 {len(chunks)} 个 chunk")

        return chunks

    # ── Step 1：逐页提取文本块 ────────────────────────────────────────

    def _extract_pages(self, pdf) -> list[dict]:
        """
        用 pdfplumber 提取每页文字块，保留 bbox 坐标和字体信息。
        返回：[{"page_num": int, "height": float, "blocks": [block, ...]}, ...]
        """
        pages = []
        for page_num, page in enumerate(pdf.pages, start=1):
            height = page.height
            blocks = []

            words = page.extract_words(
                extra_attrs=["size", "fontname"],
                use_text_flow=True,
            )
            if not words:
                continue

            # 按行分组（y0 相近的词归为同一行）
            lines = self._group_words_into_lines(words)

            for line in lines:
                text  = " ".join(w["text"] for w in line)
                y_top = min(w["top"]    for w in line)
                y_bot = max(w["bottom"] for w in line)
                sizes = [w.get("size", 0) for w in line if w.get("size")]
                font_size = max(sizes) if sizes else 0

                blocks.append({
                    "text":      text.strip(),
                    "y_top":     y_top,
                    "y_bottom":  y_bot,
                    "font_size": font_size,
                    "page_num":  page_num,
                    "page_h":    height,
                })

            pages.append({
                "page_num": page_num,
                "height":   height,
                "blocks":   blocks,
            })

        return pages

    def _group_words_into_lines(self, words: list) -> list[list]:
        """将 pdfplumber 的 word 列表按 y 坐标聚合成行。"""
        if not words:
            return []

        lines = []
        current_line = [words[0]]
        threshold = 3.0   # y 坐标差距在 3pt 内视为同一行

        for word in words[1:]:
            if abs(word["top"] - current_line[-1]["top"]) <= threshold:
                current_line.append(word)
            else:
                lines.append(current_line)
                current_line = [word]
        lines.append(current_line)
        return lines

    # ── Step 2：过滤页眉页脚 ─────────────────────────────────────────

    def _remove_headers_footers(self, pages: list) -> list:
        """
        过滤策略：
        - 位于页面顶部 8% 或底部 8% 的文本块为候选
        - 候选文本在连续 3 页以上出现相同内容（去空白比较）→ 判定为页眉/页脚
        - 全文删除这些文本
        """
        if not pages:
            return pages

        # 收集候选文本（位置在顶部/底部 8%）
        candidate_texts: Counter = Counter()
        for page in pages:
            h = page["height"]
            top_threshold    = h * 0.08
            bottom_threshold = h * 0.92
            for block in page["blocks"]:
                if block["y_top"] < top_threshold or block["y_bottom"] > bottom_threshold:
                    normalized = re.sub(r"\s+", " ", block["text"]).strip().lower()
                    if normalized:
                        candidate_texts[normalized] += 1

        # 出现次数 >= 3 的视为页眉/页脚
        header_footer_set = {
            text for text, count in candidate_texts.items()
            if count >= 3
        }

        # 从所有页面中删除
        cleaned = []
        for page in pages:
            h = page["height"]
            top_threshold    = h * 0.08
            bottom_threshold = h * 0.92
            kept_blocks = []
            for block in page["blocks"]:
                normalized = re.sub(r"\s+", " ", block["text"]).strip().lower()
                is_in_margin = (block["y_top"] < top_threshold or
                                block["y_bottom"] > bottom_threshold)
                if is_in_margin and normalized in header_footer_set:
                    continue   # 过滤
                kept_blocks.append(block)
            cleaned.append({**page, "blocks": kept_blocks})

        removed = len(header_footer_set)
        if removed:
            print(f"[Parser] 过滤 {removed} 种页眉/页脚文本")
        return cleaned

    # ── Step 3：检测正文字体大小基准 ─────────────────────────────────

    def _detect_body_font_size(self, pages: list) -> float:
        """
        统计所有文本块的字体大小，众数即为正文基准字体。
        用于后续判断标题行（字体 >= 正文 x1.2）。
        """
        sizes = []
        for page in pages:
            for block in page["blocks"]:
                if block["font_size"] > 0 and block["text"].strip():
                    sizes.append(round(block["font_size"], 1))

        if not sizes:
            return 11.0   # 默认值

        counter = Counter(sizes)
        return counter.most_common(1)[0][0]

    # ── Step 4：构建带章节标注的内容序列 ─────────────────────────────

    def _build_sections(self, pages: list, body_font_size: float) -> list[dict]:
        """
        识别章节边界，将页面内容组织成章节列表。

        章节标题判定条件（同时满足）：
        1. 字体大小 >= 正文 x1.2
        2. 行内词数 <= 12（标题通常是短句）
        3. 不是纯数字或日期行

        返回：[{"title": str, "paragraphs": [str], "pages": [int]}, ...]
        """
        heading_threshold = body_font_size * 1.2

        sections = []
        current_title = "Introduction"
        current_paras = []
        current_pages = set()
        prev_y_bottom = None

        for page in pages:
            for block in page["blocks"]:
                text = block["text"].strip()
                if not text:
                    continue

                word_count = len(text.split())
                is_heading = (
                    block["font_size"] >= heading_threshold
                    and word_count <= 12
                    and not re.fullmatch(r"[\d\s\.\-/]+", text)
                )

                if is_heading:
                    # 保存上一个章节
                    if current_paras:
                        sections.append({
                            "title":      current_title,
                            "paragraphs": current_paras,
                            "pages":      sorted(current_pages),
                        })
                    current_title = text
                    current_paras = []
                    current_pages = {block["page_num"]}
                    prev_y_bottom = None
                else:
                    # 判断是否需要开始新段落
                    # 条件：与上一行的垂直间距 > 正文行距的 1.5 倍
                    line_height = block["y_bottom"] - block["y_top"]
                    is_new_para = False
                    if prev_y_bottom is not None:
                        gap = block["y_top"] - prev_y_bottom
                        if gap > line_height * 1.5:
                            is_new_para = True

                    if is_new_para or not current_paras:
                        current_paras.append(text)
                    else:
                        current_paras[-1] += " " + text

                    current_pages.add(block["page_num"])
                    prev_y_bottom = block["y_bottom"]

        # 保存最后一个章节
        if current_paras:
            sections.append({
                "title":      current_title,
                "paragraphs": current_paras,
                "pages":      sorted(current_pages),
            })

        return sections

    # ── Step 5：过滤无关章节 ──────────────────────────────────────────

    def _filter_sections(self, sections: list) -> list:
        """
        跳过 References、IOC、About the Authors 等无 TTP 内容的章节。
        IOC 章节内容单独记录（供未来结构化处理），此处仅过滤不传给 Agent。
        """
        kept = []
        for sec in sections:
            title_lower = sec["title"].lower().strip()
            should_skip = any(
                title_lower.startswith(kw) for kw in SKIP_SECTIONS
            )
            if not should_skip:
                kept.append(sec)
        return kept

    # ── Step 6：三级切分 + Overlap ────────────────────────────────────

    def _split_and_overlap(self, sections: list) -> list[dict]:
        """
        对每个章节的段落序列执行三级切分，并添加 overlap 和元数据。
        """
        all_chunks = []
        chunk_id   = 0
        prev_words: list[str] = []   # 上一 chunk 的词列表，用于 overlap

        for sec in sections:
            title = sec["title"]
            pages = sec["pages"]

            # 把章节所有段落拼成词序列，同时记录每个词来自哪一段落
            para_word_segs = []
            for para in sec["paragraphs"]:
                words = para.split()
                if words:
                    para_word_segs.append(words)

            # 二级切分：按段落边界分块，控制词数上限
            raw_chunks = self._split_by_paragraph(para_word_segs)

            for word_list in raw_chunks:
                # 三级切分：段落块仍超限时按句子切
                sub_chunks = self._split_by_sentence(word_list)
                for sub in sub_chunks:
                    # 加 overlap：在当前 chunk 前拼接上一 chunk 末尾 100 词
                    if prev_words:
                        overlap_words_list = prev_words[-OVERLAP_WORDS:]
                        final_words = overlap_words_list + sub
                    else:
                        final_words = sub

                    text = " ".join(final_words)
                    if not text.strip():
                        continue

                    # 推算页码范围
                    page_range = (str(pages[0]) if len(pages) == 1
                                  else f"{pages[0]}-{pages[-1]}")

                    all_chunks.append({
                        "text": text,
                        "metadata": {
                            "chunk_id":   chunk_id,
                            "section":    title,
                            "page_range": page_range,
                            "word_count": len(final_words),
                            "is_first":   chunk_id == 0,
                        }
                    })

                    prev_words = sub   # 更新 overlap 缓存（用无 overlap 的原始内容）
                    chunk_id  += 1

        return all_chunks

    def _split_by_paragraph(self, para_word_segs: list) -> list[list]:
        """
        二级切分：合并段落，当累积词数超过 CHUNK_MAX_WORDS 时切割。
        尽量在段落边界切割，不在段落中间截断。
        """
        result   = []
        current  = []

        for words in para_word_segs:
            if not words:
                continue

            # 如果加入这段后超限，先保存当前块
            if current and len(current) + len(words) > CHUNK_MAX_WORDS:
                result.append(current)
                current = []

            current.extend(words)

            # 当前块已满，保存
            if len(current) >= CHUNK_MAX_WORDS:
                result.append(current)
                current = []

        if current and len(current) >= CHUNK_MIN_WORDS:
            result.append(current)
        elif current and result:
            # 末尾不足最小词数，合并到上一块
            result[-1].extend(current)

        return result if result else [current] if current else []

    def _split_by_sentence(self, words: list) -> list[list]:
        """
        三级切分：仅在块超过 CHUNK_MAX_WORDS 时启用。
        在句子边界（词以 . / ! / ? 结尾）处切割。
        """
        if len(words) <= CHUNK_MAX_WORDS:
            return [words]

        result  = []
        current = []

        for word in words:
            current.append(word)
            # 句子边界：以句号/感叹号/问号结尾
            if re.search(r"[.!?]$", word) and len(current) >= CHUNK_MIN_WORDS:
                result.append(current)
                current = []

        if current:
            if result and len(current) < CHUNK_MIN_WORDS:
                result[-1].extend(current)
            else:
                result.append(current)

        return result if result else [words]

    # ── 模拟数据（无 PDF 时的演示） ───────────────────────────────────

    def _mock_chunks(self) -> list[dict]:
        raw = [
            {"text": "Recently, a new campaign by the APT29 threat group was detected targeting research facilities.",
             "section": "Executive Summary", "page": "1"},
            {"text": "To establish a foothold, it utilized a malicious Word document containing obfuscated VBA macros.",
             "section": "Technical Analysis", "page": "2"},
            {"text": "Once executed, the malware modifies the Windows Registry Run keys to ensure it survives system reboots.",
             "section": "Technical Analysis", "page": "2"},
            {"text": "The researchers noted that the exfiltrated data was sent over an encrypted C2 channel on port 443.",
             "section": "Technical Analysis", "page": "3"},
        ]
        return [
            {
                "text": c["text"],
                "metadata": {
                    "chunk_id":   i,
                    "section":    c["section"],
                    "page_range": c["page"],
                    "word_count": len(c["text"].split()),
                    "is_first":   i == 0,
                }
            }
            for i, c in enumerate(raw)
        ]


def get_all_pdfs() -> list[Path]:
    """返回 pdf 文件夹下所有 PDF 文件列表。"""
    if not PDF_DIR.exists():
        print(f"[Warning] pdf 目录不存在：{PDF_DIR}")
        return []
    pdfs = sorted(PDF_DIR.glob("*.pdf"))
    print(f"[Parser] 发现 {len(pdfs)} 个 PDF 文件：{[p.name for p in pdfs]}")
    return pdfs


# ══════════════════════════════════════════════════════════════════════
# 2. 本地 SFT 微调模型推理引擎
# ══════════════════════════════════════════════════════════════════════

class LocalSFTModel:
    """
    本地 SFT 微调模型（标准 LoRA 全精度，20GB GPU 训练，合并后约 8GB）。
    推理运行在本机 12GB GPU 上，use_cache=True 加速自回归生成。
    """

    INSTRUCTION = "Find the techniques and ID from MITRE ATT&CK framework."

    def __init__(self, model_path: str = "./qwen3-4b-sft-merged-final-with-reasoning"):
        # 确保路径是字符串且使用正斜杠，避免 Windows 反斜杠被 transformers 误判为 Hub repo ID
        # as_posix() 把 Windows 反斜杠统一转为正斜杠，避免 transformers 校验失败
        model_path = Path(model_path).resolve().as_posix()
        print(f"\n[System] 正在加载本地微调模型: {model_path} ...")
        self.model_loaded = False
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_path, trust_remote_code=True, local_files_only=True
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True,
                local_files_only=True,
            )
            self.model.config.use_cache = True
            self.model_loaded = True
            print("[System] 本地模型加载成功！")
        except Exception as e:
            print(f"[Warning] 未检测到本地模型文件 ({e})。进入 Mock 模式。")

    def predict(self, sentence: str) -> str:
        """Alpaca 格式推理，与训练时 Prompt 严格对齐。"""
        if not self.model_loaded:
            time.sleep(0.3)
            if "VBA" in sentence or "macro" in sentence.lower():
                return "T1059.005: Visual Basic"
            if "Registry" in sentence:
                return "T1547.001: Registry Run Keys / Startup Folder"
            if "exfiltrat" in sentence.lower() or "C2" in sentence:
                return "T1041: Exfiltration Over C2 Channel"
            return "Unknown TTP"

        prompt = (
            f"### Instruction:\n{self.INSTRUCTION}\n\n"
            f"### Input:\n{sentence}\n\n"
            f"### Response:\n"
        )
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.1,
                do_sample=False,
            )
        raw = self.tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1]:],
            skip_special_tokens=True
        ).strip()
        return _parse_model_output(raw)


# ══════════════════════════════════════════════════════════════════════
# 3. 主流水线
# ══════════════════════════════════════════════════════════════════════

def run_pipeline(pdf_path: str | None = None,
                 use_stream: bool = False,
                 process_all: bool = False):
    """
    执行完整 TTP 提取流水线。

    参数：
        pdf_path:    单个 PDF 路径；None 时根据 process_all 决定
        use_stream:  True 时启用 LangGraph stream 模式，实时打印节点日志
        process_all: True 时处理 pdf/ 目录下所有 PDF
    """
    print("=" * 55)
    print("  启动基于 LangGraph + ReAct 的 TTP 提取流水线")
    print("=" * 55)

    # ── 阶段 1：初始化 ────────────────────────────────────────────────
    print("\n[阶段 1] 初始化模块...")
    print("[阶段 1] 初始化 LangGraph Agent...")
    agent       = CTILangGraphAgent()
    print("[阶段 1] 初始化本地 SFT 微调模型...")
    local_model = LocalSFTModel(
        model_path=str(
            Path(__file__).parent.parent
            .joinpath("qwen3-4b-sft-merged-final-with-reasoning")
            .resolve()
        )
    )

    # 确定要处理的 PDF 列表
    if process_all:
        pdf_list = get_all_pdfs()
    elif pdf_path:
        pdf_list = [Path(pdf_path)]
    else:
        # 没有传入 PDF，尝试从 pdf/ 目录自动读取
        pdf_list = get_all_pdfs()

    all_reports = {}

    for current_pdf in (pdf_list if pdf_list else [None]):
        _run_single(current_pdf, agent, local_model, use_stream, all_reports)

    return all_reports


def _run_single(pdf_path, agent, local_model, use_stream, all_reports):
    """处理单个 PDF 文件（或模拟数据）。"""
    label   = pdf_path.name if pdf_path else "mock_data"
    chunker = DocumentChunker(pdf_path=str(pdf_path) if pdf_path else None)

    print(f"\n{'─' * 55}")
    print(f"  处理：{label}")
    print(f"{'─' * 55}")

    # ── 阶段 2：PDF 解析 ──────────────────────────────────────────────
    chunks = chunker.get_chunks()
    print(f"[阶段 2] 解析完成，共 {len(chunks)} 个 chunk")

    # ── 阶段 3：LangGraph Agent 阅读 ─────────────────────────────────
    high_value_sentences = []
    # 从文件加载历史记忆，支持跨文档累积（首次运行时为空）
    current_summary  = load_memory()
    starting_summary = current_summary   # 保存初始状态，用于最后打印增量
    # 跨 Chunk 去重日志：已提取句子的前 12 词摘要，注入 Prompt 防止重复提取
    current_extracted_log: list = []
    trace_records:          list = []   # 每个 chunk 的工具调用日志

    print(f"\n[阶段 3] Agent 开始处理...")
    print("-" * 55)

    for chunk in chunks:
        text     = chunk["text"]
        metadata = chunk["metadata"]
        chunk_id = metadata["chunk_id"]
        section  = metadata["section"]

        print(f"\n>>> Chunk {chunk_id + 1}  [{section}]  "
              f"({metadata['word_count']} 词，第 {metadata['page_range']} 页)")

        if use_stream:
            result = {}
            for event in agent.graph.stream({
                "chunk_text":         text,
                "chunk_metadata":     metadata,
                "previous_summary":   current_summary,
                "extracted_log":      current_extracted_log,
                "messages":           [],
                "iteration_count":    0,
                "extracted_sentence": None,
                "new_summary":        current_summary,
            }):
                for node_name, node_output in event.items():
                    print(f"    [Graph] [{node_name}]")
                    result.update(node_output)
            # stream 模式下同步更新 extracted_log
            if result.get("extracted_log"):
                current_extracted_log = result["extracted_log"]
        else:
            result = agent.process_chunk(
                chunk_text=text,
                previous_summary=current_summary,
                chunk_metadata=metadata,
                extracted_log=current_extracted_log,
            )

        extracted              = result.get("extracted_sentence")
        current_summary        = result.get("new_summary", current_summary)
        current_extracted_log  = result.get("extracted_log", current_extracted_log)

        # 工具调用日志：实时打印 + 追加到 trace_records
        chunk_tool_logs = result.get("tool_log", [])
        if chunk_tool_logs:
            _print_tool_log(chunk_tool_logs[-1])
            trace_records.extend(chunk_tool_logs)

        if extracted:
            # 拆分多句话为单句列表
            sentences = _split_extracted(extracted)
            # 第二层兜底：精确字符串去重，防止极少数完全相同的句子漏网
            new_sentences = [s for s in sentences if s not in high_value_sentences]
            dedup_count   = len(sentences) - len(new_sentences)
            if dedup_count:
                print(f"    [去重] 精确匹配过滤 {dedup_count} 条重复句子")
            print(f"    [判定] 发现 TTP 行为 → 入库（{len(new_sentences)} 条）")
            high_value_sentences.extend(new_sentences)
        else:
            print(f"    [判定] 无明确 TTP 行为，过滤")

    print("-" * 55)
    print(f"\n[阶段 3] 粗筛完毕，浓缩出 {len(high_value_sentences)} 个高价值句子")

    # 持久化本次运行后的最终记忆，供下次运行加载
    save_memory(current_summary)
    show_memory_diff(starting_summary, current_summary)

    # 写入 Agent 工具调用 trace 文件
    _save_trace(label, trace_records)

    # ── 阶段 4：本地 SFT 模型精准映射 ────────────────────────────────
    print("\n[阶段 4] 本地 SFT 模型进行精准 ATT&CK 映射...")
    final_report = []
    for sentence in tqdm(high_value_sentences, desc="本地 GPU 推理"):
        mapped_ttp = local_model.predict(sentence)
        final_report.append({
            "extracted_sentence":   sentence,
            "mitre_attack_mapping": mapped_ttp,
        })

    # ── 阶段 5：输出报告 ──────────────────────────────────────────────
    output_dir = Path(__file__).parent / "data/output"
    output_dir.mkdir(parents=True, exist_ok=True)

    safe_name   = re.sub(r"[^\w]", "_", label.replace(".pdf", ""))
    report_path = output_dir / f"ttp_report_{safe_name}.json"

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(final_report, f, ensure_ascii=False, indent=4)

    print(f"\n✅ [{label}] 处理完毕！报告已保存至：{report_path}")
    print("\n--- 报告预览 ---")
    print(json.dumps(final_report, indent=4, ensure_ascii=False))

    all_reports[label] = final_report


if __name__ == "__main__":
    # ── 使用示例 ──────────────────────────────────────────────────────
    # 1. 处理 pdf/ 目录下所有 PDF（推荐）
    #run_pipeline(process_all=True, use_stream=False)

    # 2. 处理单个 PDF
    # run_pipeline(pdf_path="../pdf/apt29_report.pdf")

    # 3. 使用模拟数据（无 PDF 时演示）
    # run_pipeline()

    # 4. 开启 stream 模式，实时看到每个 LangGraph 节点的执行
    run_pipeline(process_all=True, use_stream=True)