"""
agent_core.py
-------------
CTI LangGraph Agent：用 LangGraph 管理宏观流程，用 ReAct 驱动 LLM 微观推理。

图结构：
    START
      ↓
    parse_chunk        # 纯 Python 节点：注入记忆，初始化 State
      ↓
    react_agent_node   # ReAct 节点：LLM 推理 + 按需调用 Hybrid RAG Tool
      ↓
    should_continue    # 条件边：有结果/超限 → update_memory
      ↓
    update_memory      # 纯 Python 节点：更新结构化 JSON 记忆
      ↓
    END

LangGraph 负责宏观流程控制（节点间状态显式传递、条件转移、最大迭代保护）。
ReAct 负责节点内的微观推理（Thought → Action → Observation 循环）。
"""

import os
import json
from typing import Optional, Annotated
import operator
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import create_react_agent
from typing_extensions import TypedDict

from rag_tool import search_mitre_attack_tool

load_dotenv()
QWEN_API_KEY  = os.environ.get("DASHSCOPE_API_KEY")
QWEN_BASE_URL = os.environ.get("DASHSCOPE_API_URL",
                               "https://dashscope.aliyuncs.com/compatible-mode/v1")

MAX_ITERATIONS = 3   # ReAct 最大迭代轮数，防止死循环耗尽 Token


# ══════════════════════════════════════════════════════════════════════
# 1. State 定义
#    TypedDict 强类型，每个字段的含义和归属节点都有注释
# ══════════════════════════════════════════════════════════════════════

class CTIAgentState(TypedDict):
    # ── parse_chunk 写入，后续只读 ────────────────────────────────────
    chunk_text:        str           # 当前 chunk 原文
    chunk_metadata:    dict          # section、page_range 等元数据
    previous_summary:  str           # 上一轮结构化 JSON 记忆

    # ── 跨 Chunk 全局去重日志（只增不减）────────────────────────────────
    # 存储已确认入库的 TTP 句子的前 N 词（摘要形式），注入 Prompt 让 Agent
    # 提取时主动跳过语义重叠的内容，从源头防止 overlap 区域产生重复提取
    # Annotated[list, operator.add] 保证多次调用时追加而非覆盖
    extracted_log:     Annotated[list, operator.add]

    # ── ReAct 节点读写 ────────────────────────────────────────────────
    # Annotated[list, operator.add] 告诉 LangGraph 合并时做追加而非覆盖
    # 这样 ReAct 子图产生的消息能正确累积到外层 State
    messages:          Annotated[list, operator.add]
    iteration_count:   int           # 当前迭代次数

    # ── update_memory 写入 ────────────────────────────────────────────
    extracted_sentence: Optional[str]  # 提取到的 TTP 句子，None 表示无
    new_summary:        str            # 更新后的结构化记忆

    # ── 工具调用日志（每个 chunk 的 RAG 调用记录）────────────────────────
    # 每条记录对应一次工具调用，包含查询词和召回文档列表
    # operator.add 保证跨 chunk 追加，不覆盖历史记录
    tool_log:           Annotated[list, operator.add]


# ══════════════════════════════════════════════════════════════════════
# 2. 节点函数
# ══════════════════════════════════════════════════════════════════════

def parse_chunk(state: CTIAgentState) -> dict:
    """
    【节点1：parse_chunk】纯 Python，不调用 LLM。

    职责：
    - 将 chunk 原文和结构化记忆组装成 ReAct 节点的初始消息列表
    - 重置迭代计数器
    - 注入 section 元数据，让 Agent 知道当前处理的文档章节

    LangGraph 注意点：返回 dict，只包含需要更新的字段，
    未返回的字段保持原值不变。
    """
    section     = state["chunk_metadata"].get("section", "未知章节")
    extracted_log = state.get("extracted_log", [])

    # 把已提取句子的前 12 词拼成列表，注入 Prompt 让 Agent 主动跳过重复内容
    # 只取前 12 词是为了控制 Prompt 长度，同时足够让 LLM 识别语义重叠
    already_extracted = ""
    if extracted_log:
        items = "\n".join(f"- {s}" for s in extracted_log)
        already_extracted = f"""
【已提取句子（语义重叠的内容不要再次提取）】
{items}
"""

    system_content = f"""你是一位顶级的网络安全威胁情报（CTI）分析专家。

【任务】
阅读下方威胁情报文本，找出所有明确描述攻击行为的原文句子（对应 MITRE ATT&CK 战术或技术）。

【全局上下文记忆】（用于消解"它"、"该组织"等代词的指代）：
{state['previous_summary']}

【当前文档章节】：{section}
{already_extracted}
【输出格式——必须严格遵守，违反则视为错误】
1. 每行输出一个原文句子，直接复制原文，不得修改任何词语
2. 不得输出任何解释、分析、编号、符号、Markdown 格式或注释
3. 多个句子时每句占一行，行与行之间无空行
4. 如果文本中不包含任何攻击行为，只输出一个单词：None

【句子完整性要求——违反则不提取】
- 必须是语法上完整的句子，有主语和谓语
- 不得以 including、such as、as well as、and、but、which 等词开头
- 不得以逗号结尾（说明句子被截断，后面还有内容）

【判断标准——必须同时满足以下两条才能提取】
条件一：句子必须包含具体的攻击手法，即明确说明用了什么工具、执行了什么操作、建立了什么连接
  ✅ 合格：APT39 used Mimikatz to dump credentials（有具体工具 + 具体操作）
  ✅ 合格：the group exploited vulnerable web servers to install web shells（有具体操作）
  ❌ 不合格：APT39 suggests intent to perform surveillance operations（只有意图，没有具体手法）
  ❌ 不合格：APT39 focuses on collecting personal information（只有目的，没有具体手法）
  ❌ 不合格：targeting data supports the belief that APT39's mission is to track targets（分析性描述）

条件二：句子不属于以下任何一类（即使它和攻击相关）
  - 意图 / 目的描述：suggests intent to、focuses on、aims to、in order to、for the purpose of
  - 战略 / 国家利益：national priorities、strategic requirements、geopolitical
  - 组织背景描述：APT39 was created、activities largely align with、considered distinct from
  - 目标行业描述：has prioritized the telecommunications sector、targeting of travel industry
  - 分析推断语句：we believe、supports the belief that、indicates that、reflects efforts to

不确定某行为对应哪个 ATT&CK 技术时，调用知识库工具辅助判断

【合格句子的主语类型——以下主语均可接受】
- 威胁组织：APT29 leveraged...、The group deployed...、The attackers activated...
- 恶意软件：The malware establishes...、The trojan decrypts...、The backdoor connects...
- 恶意代码：The macro drops...、The loader executes...、The shellcode injects...
- 被动语态：Credentials were dumped using...、A scheduled task was created to...

【输出示例】
正确示例（以下所有句式均合格）：
APT29 leveraged spear phishing emails with malicious attachments to deliver POWBAT.
The group used Mimikatz to dump credentials from LSASS memory.
The attackers deployed NBTscan to identify Windows hosts with NetBIOS services enabled.
The macro establishes scheduled tasks to ensure persistence on the victim host.
The malware activates a Python-based HTTP server to upload and execute follow-up components.
The loader trojan DangerAds executes malicious code only if the victim's username matches predefined strings.
The attackers used DLL side-loading to execute a malicious payload via a legitimate process.
A custom backdoor named AtlasAgent was implanted to establish communication with CnC servers.

错误示例（以下类型禁止输出）：
APT39's focus on telecommunications suggests intent to perform surveillance.
We believe APT39 targets personal information to support national priorities.
The group's activities reflect efforts to collect geopolitical data for nation-state decision making.
including the use of Mimikatz to dump credentials,"""

    return {
        "messages":        [SystemMessage(content=system_content),
                            HumanMessage(content=state["chunk_text"])],
        "iteration_count": 0,
        "extracted_sentence": None,
    }



def _parse_final_answer(raw: str) -> str | None:
    """
    从 ReAct 输出中提取 Final Answer 部分，过滤掉推理过程。

    处理三种输出格式：
    1. 包含 "Final Answer:" 标记  → 取标记之后的内容
    2. 直接是干净的句子          → 直接使用
    3. "None" / 空               → 返回 None

    多句话处理：
    如果 Final Answer 里包含多个句子（以 - 开头的列表，或换行分隔），
    序列化为 JSON 数组字符串，main.py 里负责拆开逐条传给本地模型。
    """
    import re as _re

    if not raw:
        return None

    text = raw.strip()

    # ── Step 1：找 Final Answer 标记 ──────────────────────────────────
    # 兼容各种格式：Final Answer:、Final answer:、**Final Answer:**
    fa_pattern = _re.compile(
        r"(?:^|\n)\*{0,2}[Ff]inal\s+[Aa]nswer\*{0,2}\s*:+\s*",
        _re.MULTILINE
    )
    match = fa_pattern.search(text)
    if match:
        text = text[match.end():].strip()

    # ── Step 2：判断是否为 None ───────────────────────────────────────
    if not text or text.lower().rstrip(".!") in ("none", "null", "no ttp", "no ttp found"):
        return None

    # ── Step 3：提取多句话 ────────────────────────────────────────────
    sentences = _extract_sentences(text)

    if not sentences:
        return None
    if len(sentences) == 1:
        return sentences[0]

    # 多句话序列化为 JSON 数组，main.py 里负责拆开
    import json as _json
    return _json.dumps(sentences, ensure_ascii=False)


def _extract_sentences(text: str) -> list[str]:
    """
    从文本中提取 TTP 句子列表。

    支持的格式：
    - 带 "-" 前缀的列表项
    - 带数字编号的列表项（1. 2. 等）
    - 带 ** 加粗的句子
    - 普通换行分隔的句子
    """
    import re as _re

    lines = [l.strip() for l in text.split("\n") if l.strip()]
    sentences = []

    for line in lines:
        # 去掉列表前缀：- "句子" 或 1. 句子 或 **句子**
        line = _re.sub(r"^[-\*\d]+[\.)\s]+", "", line).strip()
        line = _re.sub(r"^\*{1,2}|\*{1,2}$", "", line).strip()
        line = line.strip('"').strip("'").strip()

        # 跳过：空行、纯数字、过短的行（< 20 字符）、注释性文字
        if not line or len(line) < 20:
            continue
        skip_prefixes = (
            "this sentence", "the sentence", "note:", "however",
            "but ", "so ", "thus", "therefore", "final answer",
            "→", "maps to", "aligns with", "corresponds to",
        )
        if any(line.lower().startswith(p) for p in skip_prefixes):
            continue

        sentences.append(line)

    return sentences


def build_react_node(llm, tools):
    """
    构建 ReAct 节点闭包。

    使用 LangGraph 内置的 create_react_agent 创建 ReAct 子图。
    子图内部处理 Thought/Action/Observation 循环，
    外层 LangGraph 图通过 State 与它交互。

    LangGraph 注意点：
    create_react_agent 返回的是一个可调用的 CompiledGraph，
    在节点函数里用 .invoke() 调用，传入 messages，取出结果。
    """
    react_subgraph = create_react_agent(
        model=llm,
        tools=tools,
    )

    def react_agent_node(state: CTIAgentState) -> dict:
        """
        【节点2：react_agent_node】ReAct 推理节点。

        将外层 State 的 messages 传入 ReAct 子图，
        子图内部执行 Thought → (Action → Observation)* → Final Answer，
        最终消息写回外层 State。

        输出解析规则：
        1. 优先提取 "Final Answer:" 之后的内容，去掉推理过程
        2. 如果包含多个句子（换行分隔），拆成列表存入 extracted_sentence
           格式：JSON 数组字符串 ["句子1", "句子2"]，main.py 里再拆开逐条处理
        3. 内容为 None / 空 → extracted_sentence = None
        """
        result = react_subgraph.invoke({"messages": state["messages"]})

        # 取 ReAct 子图产生的最终回复
        final_msg     = result["messages"][-1]
        final_content = (final_msg.content
                         if hasattr(final_msg, "content")
                         else str(final_msg))

        extracted = _parse_final_answer(final_content)

        return {
            "messages":           result["messages"],
            "extracted_sentence": extracted,
            "iteration_count":    state["iteration_count"] + 1,
        }

    return react_agent_node


def _is_ai_message(msg) -> bool:
    class_name = type(msg).__name__
    return "AI" in class_name or "Assistant" in class_name


def _is_tool_message(msg) -> bool:
    class_name = type(msg).__name__
    return "Tool" in class_name or hasattr(msg, "tool_call_id")


def _parse_tool_calls(messages: list, chunk_metadata: dict) -> dict:
    """
    从 ReAct 节点产生的 messages 列表中提取工具调用记录。
    不依赖类名字符串完全匹配，直接检查 tool_calls 属性，
    兼容不同版本的 LangChain/LangGraph。
    """
    import re as _re

    tool_records = []
    call_index   = 0

    for i, msg in enumerate(messages):
        tool_calls_attr = getattr(msg, "tool_calls", None)
        if not tool_calls_attr:
            continue

        for tc in tool_calls_attr:
            call_index += 1
            query = ""
            if isinstance(tc, dict):
                query = tc.get("args", {}).get("query", "") or tc.get("query", "")
            elif hasattr(tc, "args"):
                args = tc.args
                if isinstance(args, dict):
                    query = args.get("query", "")

            recalled_docs = []
            for j in range(i + 1, len(messages)):
                next_msg = messages[j]
                if _is_tool_message(next_msg):
                    raw_result = getattr(next_msg, "content", "") or ""
                    blocks = raw_result.split("\n---\n")
                    for rank, block in enumerate(blocks[:3], start=1):
                        id_match   = _re.search(r"TTP ID:\s*(T[\d\.]+)", block)
                        name_match = _re.search(r"名称:\s*(.+)", block)
                        recalled_docs.append({
                            "rank":   rank,
                            "ttp_id": id_match.group(1) if id_match else "Unknown",
                            "name":   name_match.group(1).strip() if name_match else "Unknown",
                        })
                    break
                elif getattr(next_msg, "tool_calls", None):
                    break

            tool_records.append({
                "call_index":    call_index,
                "query":         query,
                "recalled_docs": recalled_docs,
            })

    return {
        "chunk_id":       chunk_metadata.get("chunk_id", -1),
        "section":        chunk_metadata.get("section", ""),
        "page_range":     chunk_metadata.get("page_range", ""),
        "tool_called":    len(tool_records) > 0,
        "tool_call_count": len(tool_records),
        "tool_calls":     tool_records,
    }


def build_update_memory_node(summary_chain):
    """
    构建 update_memory 节点闭包。

    结构化记忆更新规则：
    - 只增不减：threat_actor、malware_names、targets 只追加新实体
    - techniques_seen 去重追加
    - try-catch 保底：JSON 解析失败时保留旧记忆，防止记忆雪崩
    """
    def update_memory(state: CTIAgentState) -> dict:
        """【节点3：update_memory】更新结构化 JSON 记忆 + 追加 extracted_log。"""
        # ── 更新结构化记忆 ────────────────────────────────────────────
        try:
            response = summary_chain.invoke({
                "old_summary": state["previous_summary"],
                "new_chunk":   state["chunk_text"],
                "extracted":   state["extracted_sentence"] or "无",
            })
            new_summary = response.content
            json.loads(new_summary)   # 验证 JSON 合法性
        except Exception:
            new_summary = state["previous_summary"]

        # ── 把新提取的句子摘要追加到 extracted_log ───────────────────
        # 只存前 12 词作为摘要，足够让 Agent 识别语义重叠，同时控制 Prompt 长度
        new_log_entries = []
        extracted = state.get("extracted_sentence")
        if extracted:
            try:
                sentences = json.loads(extracted) if extracted.startswith("[") else [extracted]
            except Exception:
                sentences = [extracted]
            for sent in sentences:
                if sent and isinstance(sent, str):
                    # 20 词摘要：让 Agent 有足够上下文识别截断版重复句
                    snippet = " ".join(sent.split()[:20])
                    new_log_entries.append(snippet)

        # ── 解析工具调用日志 ─────────────────────────────────────────
        tool_log_entry = _parse_tool_calls(
            state.get("messages", []),
            state.get("chunk_metadata", {})
        )
        # 追加本 chunk 的工具调用记录（tool_log 用 operator.add，会跨 chunk 累积）
        new_tool_log = [tool_log_entry]

        return {
            "new_summary":   new_summary,
            "extracted_log": new_log_entries,
            "tool_log":      new_tool_log,
        }

    return update_memory


# ══════════════════════════════════════════════════════════════════════
# 3. 条件边
# ══════════════════════════════════════════════════════════════════════

def should_continue(state: CTIAgentState) -> str:
    """
    【条件边：should_continue】

    LangGraph 的核心优势：显式定义流程转移条件，
    而不是像 ReAct AgentExecutor 那样由 LLM 隐式控制停止时机。

    路由逻辑（三条路径均指向 update_memory）：
    1. 已提取到结果     → update_memory（正常完成）
    2. 明确无 TTP       → update_memory（正常完成，结果为 None）
    3. 超过最大迭代次数 → update_memory（防死循环强制终止）
    """
    if state["iteration_count"] >= MAX_ITERATIONS:
        return "update_memory"

    last_msg = state["messages"][-1] if state["messages"] else None
    if last_msg and hasattr(last_msg, "content"):
        content = last_msg.content.strip()
        if "Final Answer" in content or content.lower() in ("none", "none."):
            return "update_memory"

    # 兜底：已执行过 ReAct 就往下走
    if state["iteration_count"] > 0:
        return "update_memory"

    return "update_memory"


# ══════════════════════════════════════════════════════════════════════
# 4. CTILangGraphAgent：图的组装与对外接口
# ══════════════════════════════════════════════════════════════════════

class CTILangGraphAgent:
    """
    基于 LangGraph + ReAct 的 CTI TTP 提取 Agent。

    两层控制分离：
    - LangGraph 管理宏观流程（节点间状态显式传递、条件转移边、迭代保护）
    - ReAct 驱动微观推理（Thought/Action/Observation 循环、工具调用决策）
    """

    def __init__(self):
        # LLM：低温度保证推理严谨性
        self.llm = ChatOpenAI(
            model="qwen-plus-latest",
            temperature=0.1,
            api_key=QWEN_API_KEY,
            base_url=QWEN_BASE_URL,
        )

        self.tools = [search_mitre_attack_tool]

        # 结构化记忆更新链
        # 注意：Prompt 要求 LLM 只输出 JSON，不输出任何额外文字
        from langchain_core.prompts import PromptTemplate
        summary_prompt = PromptTemplate.from_template(
            """基于之前的结构化记忆和最新阅读的文本，更新威胁情报记忆。

规则：
1. 只增不减：已有实体不删除，只追加新发现的实体
2. techniques_seen 只记录具体的 MITRE ATT&CK 攻击技术或行为描述（如 spear phishing、Registry persistence）
   - 禁止写入章节标题（如 Initial Compromise、Lateral Movement、Establish Foothold）
   - 禁止写入组织名称、目标行业、意图描述
   - 只从"本轮提取结果"字段中提取技术内容，不从"最新文本"中直接提取
3. 去重：已有内容不重复写入
4. 只输出 JSON，不输出任何解释文字

之前的记忆（JSON）：{old_summary}
最新文本：{new_chunk}
本轮提取结果（只从此字段提取 techniques_seen）：{extracted}

输出更新后的 JSON（格式与输入保持一致）："""
        )
        self.summary_chain = summary_prompt | self.llm

        # 构建节点函数
        react_node    = build_react_node(self.llm, self.tools)
        memory_node   = build_update_memory_node(self.summary_chain)

        # 构建并编译 LangGraph 图
        self.graph = self._build_graph(react_node, memory_node)

        print("✅ CTILangGraphAgent 初始化完成")
        print("   图结构：parse_chunk → react_agent_node → [条件边] → update_memory → END")

    def _build_graph(self, react_node, memory_node) -> object:
        """组装 StateGraph，注册节点和边，编译为可执行图。"""
        graph = StateGraph(CTIAgentState)

        # 注册节点
        graph.add_node("parse_chunk",      parse_chunk)
        graph.add_node("react_agent_node", react_node)
        graph.add_node("update_memory",    memory_node)

        # 入口
        graph.set_entry_point("parse_chunk")

        # 固定边：parse_chunk → react_agent_node
        graph.add_edge("parse_chunk", "react_agent_node")

        # 条件边：react_agent_node → should_continue → update_memory
        # 显式定义转移条件，比 AgentExecutor.max_iterations 更可控
        graph.add_conditional_edges(
            "react_agent_node",
            should_continue,
            {"update_memory": "update_memory"},
        )

        # 固定边：update_memory → END
        graph.add_edge("update_memory", END)

        return graph.compile()

    def process_chunk(self,
                      chunk_text: str,
                      previous_summary: str,
                      chunk_metadata: dict | None = None,
                      extracted_log: list | None = None) -> dict:
        """
        处理单个 chunk，返回提取结果和更新后的记忆。

        参数：
            chunk_text:       当前 chunk 原文
            previous_summary: 上一轮结构化 JSON 记忆字符串
            chunk_metadata:   可选，包含 section、page_range 等信息
            extracted_log:    已提取句子的前 12 词摘要列表，用于跨 chunk 去重

        返回：
            {
                "extracted_sentence": str | None,  # 提取到的 TTP 句子
                "new_summary":        str,          # 更新后的记忆
                "extracted_log":      list,         # 追加后的去重日志
            }
        """
        if chunk_metadata is None:
            chunk_metadata = {}
        if extracted_log is None:
            extracted_log = []

        initial_state: CTIAgentState = {
            "chunk_text":         chunk_text,
            "chunk_metadata":     chunk_metadata,
            "previous_summary":   previous_summary,
            "extracted_log":      extracted_log,
            "messages":           [],
            "iteration_count":    0,
            "extracted_sentence": None,
            "new_summary":        previous_summary,
            "tool_log":           [],
        }

        try:
            result = self.graph.invoke(initial_state)
            return {
                "extracted_sentence": result["extracted_sentence"],
                "new_summary":        result["new_summary"],
                "extracted_log":      result.get("extracted_log", []),
                "tool_log":           result.get("tool_log", []),
            }
        except Exception as e:
            print(f"Agent 处理出错: {e}")
            return {
                "extracted_sentence": None,
                "new_summary":        previous_summary,
            }

    def stream_chunk(self,
                     chunk_text: str,
                     previous_summary: str,
                     chunk_metadata: dict | None = None):
        """
        流式处理，实时打印每个节点的输出，适合调试和演示。

        用法：
            for event in agent.stream_chunk(chunk, summary):
                pass   # 节点输出已在内部打印
        """
        if chunk_metadata is None:
            chunk_metadata = {}

        initial_state: CTIAgentState = {
            "chunk_text":         chunk_text,
            "chunk_metadata":     chunk_metadata,
            "previous_summary":   previous_summary,
            "messages":           [],
            "iteration_count":    0,
            "extracted_sentence": None,
            "new_summary":        previous_summary,
        }

        final_result = None
        for event in self.graph.stream(initial_state):
            for node_name, node_output in event.items():
                print(f"\n  [LangGraph] 节点 [{node_name}] 执行完毕")
                if node_name == "react_agent_node":
                    extracted = node_output.get("extracted_sentence")
                    print(f"    提取结果: {extracted or '(无 TTP)'}")
                elif node_name == "update_memory":
                    print(f"    记忆已更新")
                    final_result = node_output

        return final_result


# ── 测试代码 ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("--- 正在初始化 CTILangGraphAgent ---")
    agent = CTILangGraphAgent()

    # 初始化结构化记忆
    initial_memory = json.dumps({
        "threat_actor":    "",
        "malware_names":   [],
        "targets":         [],
        "techniques_seen": []
    }, ensure_ascii=False)

    chunk_1 = "The notorious APT29 group has initiated a new campaign targeting healthcare organizations."
    chunk_2 = "To maintain access, it modified the Windows Registry Run keys to execute its payload upon system boot."

    print("\n--- [处理 Chunk 1] ---")
    res1 = agent.process_chunk(
        chunk_1,
        previous_summary=initial_memory,
        chunk_metadata={"section": "Executive Summary"}
    )
    print(f"提取结果: {res1['extracted_sentence']}")
    print(f"更新记忆: {res1['new_summary']}")

    print("\n--- [处理 Chunk 2，携带 Chunk 1 的记忆] ---")
    res2 = agent.process_chunk(
        chunk_2,
        previous_summary=res1["new_summary"],
        chunk_metadata={"section": "Technical Analysis"}
    )
    print(f"提取结果: {res2['extracted_sentence']}")
    print(f"更新记忆: {res2['new_summary']}")