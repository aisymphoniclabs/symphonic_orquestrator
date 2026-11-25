"""
Continue.dev LangGraph Proxy — Workflow AI Ollama (v0.5.0)
Arquitetura Dual-Model com Multi-Tool Chaining (`return_direct=False` + Append por `msg.name`)
e Sumarização Técnica Detalhada no Helper (`LFM2.5-1.2B`):
- Helper LLM (`liquid/lfm2.5-1.2b` / `lfm2.5-helper`): Sumarização incremental detalhada em 4 dimensões
  (Tópico/Objetivo, Código/Arquivos Desenvolvidos, Stack/Decisões/Restrições e Estado Atual para turnos futuros)
  via `.invoke()` direto com janela de leitura ampliada (`[:1800]` chars por mensagem).
- Main LLM (`qwen/qwen3-1.7b` / `qwen3-main`):
  1. Ciclo Multi-Tool no `create_agent` (`return_direct=False`, `recursion_limit=6`): o agente pode
     chamar uma tool (`tavily_web_search`), receber e avaliar seu `ToolMessage`, encadear outra tool
     (`sandbox_file_manager`) se necessário, e anexar todos os resultados diferenciados por `msg.name`.
  2. Validação AST (`ast.parse`) no `write_file` + síntese limpa via `.invoke()` (aproveitando docs já
     buscados pelo Tavily no mesmo turno) e modo de explicação sem repetir o código inteiro no chat.
"""

import ast
import os
import re
import json
import time
import uuid
from pathlib import Path
from typing import Dict, List, Literal, Set, TypedDict, Optional, Any
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn
from tavily import TavilyClient
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, BaseMessage
from langchain.tools import tool
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

try:
    from langchain.agents import create_agent
except ImportError:
    from langgraph.prebuilt import create_react_agent as create_agent

# ==========================================
# 1. CONFIGURAÇÃO DE AMBIENTE, SANDBOX E API
# ==========================================

load_dotenv()

SANDBOX_DIR = Path(__file__).resolve().parent / "sandbox"
SANDBOX_DIR.mkdir(parents=True, exist_ok=True)

LOCAL_LLM_BASE_URL = os.environ.get("LOCAL_LLM_BASE_URL")
LOCAL_LLM_API_KEY = os.environ.get("LOCAL_LLM_API_KEY")
HELPER_MODEL_NAME = os.environ.get("HELPER_MODEL_NAME")
MAIN_MODEL_NAME = os.environ.get("MAIN_MODEL_NAME")
# TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY")
TAVILY_API_KEY = ""

app = FastAPI(title="Symphonic AI Orquestrator", version="0.5.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==========================================
# 2. CONFIGURAÇÃO DOS MODELOS (SEM OVERRIDES)
# ==========================================
helper_llm = ChatOpenAI(
    base_url=LOCAL_LLM_BASE_URL,
    api_key=LOCAL_LLM_API_KEY,
    model=HELPER_MODEL_NAME,
)

main_llm = ChatOpenAI(
    base_url=LOCAL_LLM_BASE_URL,
    api_key=LOCAL_LLM_API_KEY,
    model=MAIN_MODEL_NAME,
)

tavily_client = TavilyClient(api_key=TAVILY_API_KEY) if TAVILY_API_KEY else None

SESSION_MEMORY = {
    "summary": "",
    "summarized_count": 0,
}

CURRENT_TURN_CONTEXT: Dict[str, str] = {
    "user_prompt": "",
    "memory_summary": "",
    "web_docs_fetched": "",
}

TURN_EXECUTED_SIGNATURES: Set[str] = set()

# ==========================================
# 3. FUNÇÕES SINTÁTICAS, VALIDAÇÃO AST E TOOLS `@tool`
# ==========================================
def sanitize_text_single_newline(raw_text: str) -> str:
    if not raw_text:
        return ""
    cleaned = raw_text.replace("\r\n", "\n").replace("\r", "\n")
    cleaned = re.sub(r"<think>.*?</think>", "", cleaned, flags=re.DOTALL)
    cleaned = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f\u200b-\u200d\ufeff]", "", cleaned)
    cleaned = re.sub(r"[ \t]+", " ", cleaned)
    cleaned = re.sub(r" *\n *", "\n", cleaned)
    cleaned = re.sub(r"\n{2,}", "\n", cleaned)
    return cleaned.strip()


def sanitize_raw_tavily_snippet(raw_text: str, max_chars: int = 450) -> str:
    cleaned = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", raw_text)
    cleaned = cleaned.replace("[...]", " ")
    cleaned = re.sub(r"[\[\]{}]", "", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned[:max_chars]


def strip_outer_markdown_fences(code_text: str) -> str:
    stripped = code_text.strip()
    blocks = re.findall(r"```(?:[a-zA-Z0-9_+-]*)\n(.*?)```", stripped, flags=re.DOTALL)
    if blocks:
        return max(blocks, key=lambda b: len(b.strip())).strip()
    match = re.match(r"^```[a-zA-Z0-9_+-]*\n(.*?)\n```$", stripped, flags=re.DOTALL)
    if match:
        return match.group(1).strip()
    return stripped


def is_code_broken_or_incomplete(filename: str, code_text: str) -> bool:
    cleaned = code_text.strip()
    if not cleaned or len(cleaned) < 40:
        return True
    if re.search(r"=\s*\.\.\.|^\s*\.\.\.\s*$", cleaned, flags=re.MULTILINE):
        return True
    if filename.endswith(".py"):
        try:
            ast.parse(cleaned)
        except SyntaxError as syn_err:
            print(f"[Validador AST Python] Erro sintático no rascunho da tool: {syn_err}")
            return True
    return False


def synthesize_clean_file_code_via_invoke(filename: str) -> str:
    user_req = CURRENT_TURN_CONTEXT.get("user_prompt", "")
    mem_sum = CURRENT_TURN_CONTEXT.get("memory_summary", "")
    web_docs = CURRENT_TURN_CONTEXT.get("web_docs_fetched", "")

    web_block = f"- **Verified Web Documentation Fetched in This Turn:**\n{web_docs}\n" if web_docs else ""
    code_gen_prompt = (
        f"# TASK: GENERATE COMPLETE PRODUCTION SOURCE CODE FOR FILE '{filename}'\n"
        f"- **Prior Session Context:** {mem_sum if mem_sum else 'None'}\n"
        f"{web_block}"
        f"- **User Requirements:** {user_req}\n"
        "# STRICT CODE OUTPUT RULES\n"
        "- Output ONLY a single Markdown code block containing the complete, runnable, syntax-checked source code.\n"
        "- Use exact, verified library imports and methods (from Verified Web Documentation if present).\n"
        "- Do NOT include any conversational prose before or after the code block."
    )
    print(f"[Code Synthesizer] Gerando código completo via .invoke() livre para './sandbox/{filename}'...")
    resp = main_llm.invoke([HumanMessage(content=code_gen_prompt)])
    raw_content = resp.content if hasattr(resp, "content") else str(resp)
    raw_content = re.sub(r"<think>.*?</think>", "", raw_content, flags=re.DOTALL).strip()
    return strip_outer_markdown_fences(raw_content)


class TavilySearchInput(BaseModel):
    """Input schema for the Tavily technical documentation search tool."""
    query: str = Field(
        description=(
            "Concise English search query (3 to 6 keywords) containing the exact library, "
            "framework, and method name (e.g., 'geopy distance geodesic kilometers python docs')."
        )
    )


class SandboxFileToolInput(BaseModel):
    """Input schema for reading and writing files inside the local sandbox directory."""
    operacao: Literal["read_file", "write_file"] = Field(
        description=(
            "Action to perform: "
            "Use 'write_file' whenever the user asks to save, create, store, or write a script/file in the sandbox. "
            "Use 'read_file' whenever the user asks to read, open, or inspect an existing file in the sandbox."
        )
    )
    nome_arquivo: str = Field(
        description=(
            "Target filename with extension inside ./sandbox/ (e.g., 'distance_calculator.py', 'script.py', 'Main.kt'). "
            "If the user asks to save a file in the sandbox without naming it, invent a clear snake_case filename."
        )
    )
    conteudo: str = Field(
        default="",
        description=(
            "When operacao='write_file': provide the complete, runnable source code to save into the file "
            "(with all imports, functions, and proper indentation, without ``` fences). "
            "When operacao='read_file': pass an empty string ''."
        ),
    )


@tool("tavily_web_search", args_schema=TavilySearchInput, return_direct=False)
def tavily_web_search(query: str) -> str:
    """
    Search external web documentation for libraries, APIs, methods, or version deprecations.
    Can be chained BEFORE `sandbox_file_manager` when you need to verify library imports/methods
    before writing a file to disk.
    """
    clean_query = sanitize_text_single_newline(query)
    sig = f"tavily:{clean_query.lower()}"
    if sig in TURN_EXECUTED_SIGNATURES:
        return (
            "ALREADY_SEARCHED: You already executed this web search in this turn. "
            "Proceed to the next required tool (such as `sandbox_file_manager`) or finish with 'DONE'."
        )
    TURN_EXECUTED_SIGNATURES.add(sig)

    print(f"[Main Tool Agent] Acionou tavily_web_search: '{clean_query}'")

    if tavily_client is None:
        return "No relevant external web documentation found (TAVILY_API_KEY not configured)."

    snippets: List[str] = []
    try:
        response = tavily_client.search(
            query=clean_query,
            search_depth="basic",
            max_results=2,
            include_raw_content=False,
            include_answer=False,
        )
        results = response.get("results", [])
        filtered = [r for r in results if r.get("score", 1.0) >= 0.40][:2] or results[:2]
        for idx, item in enumerate(filtered, 1):
            title = sanitize_raw_tavily_snippet(item.get("title", "Doc"), max_chars=90)
            url = item.get("url", "")
            body = sanitize_raw_tavily_snippet(item.get("content", ""), max_chars=450)
            snippets.append(f"- **Source {idx} ({title} - {url}):** {body}")
    except Exception as exc:
        print(f"[Tavily API Error]: {exc}")

    if not snippets:
        return "No relevant external web documentation found."

    joined_docs = "\n".join(snippets)
    CURRENT_TURN_CONTEXT["web_docs_fetched"] = joined_docs
    return (
        f"WEB_SEARCH_SUCCESS for '{clean_query}':\n{joined_docs}\n"
        "EVALUATION HINT: If the user also requested saving or reading a file in the sandbox, "
        "call `sandbox_file_manager` now using these verified facts; otherwise reply 'DONE'."
    )


@tool("sandbox_file_manager", args_schema=SandboxFileToolInput, return_direct=False)
def sandbox_file_manager(
    operacao: Literal["read_file", "write_file"],
    nome_arquivo: str,
    conteudo: str = "",
) -> str:
    """
    Read or write files inside the local './sandbox/' directory.
    MUST be called whenever the user explicitly asks to save, write, create, or read a file in the sandbox.
    Do NOT call this tool if the user did not ask to save or read a file.
    """
    safe_name = os.path.basename(nome_arquivo.strip()) or "generated_script.py"
    sig = f"file:{operacao}:{safe_name}"
    if sig in TURN_EXECUTED_SIGNATURES:
        return (
            f"ALREADY_EXECUTED: `{operacao}` on `./sandbox/{safe_name}` was already completed in this turn. "
            "Do NOT call `sandbox_file_manager` again; reply 'DONE' to finish tool execution."
        )
    TURN_EXECUTED_SIGNATURES.add(sig)

    target_path = SANDBOX_DIR / safe_name
    print(
        f"[Main Tool Agent] sandbox_file_manager: operacao='{operacao}' | "
        f"nome_arquivo='{safe_name}' | len(conteudo_tool)={len(conteudo)} chars"
    )

    if operacao == "read_file":
        if not target_path.exists() or not target_path.is_file():
            return f"FILE_TOOL_ERROR: File '{safe_name}' does not exist in ./sandbox/."
        try:
            file_text = target_path.read_text(encoding="utf-8")
            return (
                f"FILE_READ_SUCCESS: Full content of './sandbox/{safe_name}' ({len(file_text)} chars):\n"
                f"```\n{file_text}\n```\n"
                "EVALUATION HINT: File read complete. If no other tool is needed, reply 'DONE'."
            )
        except Exception as exc:
            return f"FILE_TOOL_ERROR: Failed to read '{safe_name}': {exc}"

    if operacao == "write_file":
        cleaned_content = strip_outer_markdown_fences(conteudo)
        if is_code_broken_or_incomplete(safe_name, cleaned_content):
            cleaned_content = synthesize_clean_file_code_via_invoke(safe_name)

        try:
            target_path.write_text(cleaned_content + "\n", encoding="utf-8")
            size_bytes = len((cleaned_content + "\n").encode("utf-8"))
            line_count = len(cleaned_content.splitlines())
            return (
                f"FILE_WRITE_SUCCESS: File './sandbox/{safe_name}' has been saved to disk "
                f"({size_bytes} bytes, {line_count} lines).\n"
                f"SAVED_FILENAME: ./sandbox/{safe_name}\n"
                f"SAVED_CODE_REFERENCE:\n```\n{cleaned_content}\n```\n"
                "EVALUATION HINT: The file is saved on disk. Do NOT call `sandbox_file_manager` again; reply 'DONE'."
            )
        except Exception as exc:
            return f"FILE_TOOL_ERROR: Failed to write '{safe_name}': {exc}"

    return f"FILE_TOOL_ERROR: Unsupported operation '{operacao}'."


main_tool_agent = create_agent(
    model=main_llm,
    tools=[tavily_web_search, sandbox_file_manager],
)


class OpenAIMessage(BaseModel):
    role: str
    content: Any


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[OpenAIMessage]
    stream: Optional[bool] = True
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None


# ==========================================
# 4. PREPARAÇÃO DE CONTEÚDO E MONTAGEM DE PROMPT
# ==========================================
def normalize_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            [item.get("text", "") for item in content if isinstance(item, dict) and "text" in item]
        )
    return str(content)


def sanitize_user_prompt(raw_text: str) -> str:
    if not raw_text:
        return ""
    cleaned = raw_text.replace("\r\n", "\n").replace("\r", "\n")
    cleaned = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f\u200b-\u200d\ufeff]", "", cleaned)
    lines = [line.rstrip() for line in cleaned.splitlines()]
    normalized_lines = []
    for line in lines:
        if line and not line.startswith((" ", "\t")):
            normalized_lines.append(" ".join(line.split()))
        else:
            normalized_lines.append(line)
    return re.sub(r"\n{2,}", "\n", "\n".join(normalized_lines)).strip()


def assemble_final_prompt(
    memoria: str,
    ide_prompt: str,
    dados_web: str,
    prompt_usuario: str,
) -> List[BaseMessage]:
    blocks: List[str] = []

    if ide_prompt:
        blocks.append(f"# IDE CONTEXT RULES\n{sanitize_text_single_newline(ide_prompt)}")

    if memoria and memoria != "Nenhum histórico.":
        blocks.append(
            "# PRIOR SESSION SUMMARY (BACKGROUND ONLY - DO NOT ANSWER PAST TURNS)\n"
            f"{sanitize_text_single_newline(memoria)}"
        )

    has_tool_data = dados_web and dados_web not in (
        "Nenhuma busca web solicitada.",
        "No relevant external web documentation found.",
    )
    is_file_write_turn = bool(has_tool_data and "FILE_WRITE_SUCCESS:" in dados_web)

    if has_tool_data:
        blocks.append(
            "# TOOL EXECUTION RESULTS (APPENDED BY TOOL NAME - USE TO GROUND YOUR ANSWER)\n"
            f"{dados_web.strip()}"
        )

    if is_file_write_turn:
        blocks.append(
            "# CURRENT USER REQUEST (POST-FILE-WRITE EXPLANATION & EXECUTION GUIDE MODE)\n"
            f"Original Request: {prompt_usuario}\n"
            "# MANDATORY RESPONSE INSTRUCTIONS FOR SAVED FILE\n"
            "- The complete source code has ALREADY been saved to `./sandbox/` (see `SAVED_FILENAME` and `SAVED_CODE_REFERENCE` above).\n"
            "- Do NOT rewrite or repeat the full source code block in the chat.\n"
            "- Instead, structure your response clearly with:\n"
            "  1. **Saved File Confirmation & Architecture:** Confirm the exact file path saved in `./sandbox/` and explain how the implemented code works (libraries, functions, formulas, and logic).\n"
            "  2. **How to Run & Dependencies:** Show the exact terminal commands to install any required packages (e.g., `pip install ...`) and execute the saved script.\n"
            "  3. **Next Possibilities & Extensions:** Suggest 2 to 3 practical improvements or features the user can add next."
        )
    else:
        blocks.append(
            "# CURRENT USER REQUEST (YOUR ACTIVE REQUEST TO BE ANSWERED AND FOCUSED FOR THE USER)\n"
            f"{prompt_usuario}"
        )

    return [HumanMessage(content="\n".join(blocks))]


# ==========================================
# 5. DEFINIÇÃO DO ESTADO E NÓS DO LANGGRAPH
# ==========================================
class AgentState(TypedDict):
    messages: List[BaseMessage]
    system_prompt_ide: str
    short_term_memory: str
    clean_user_prompt: str
    web_search_data: str
    final_prompt: List[BaseMessage]


def memory_and_input_prep_node(state: AgentState):
    messages = state.get("messages", [])
    total_msgs = len(messages)

    if total_msgs <= 2:
        SESSION_MEMORY["summary"] = ""
        SESSION_MEMORY["summarized_count"] = 0

    current_summary = SESSION_MEMORY["summary"]
    summarized_count = SESSION_MEMORY["summarized_count"]

    if total_msgs > 3:
        cutoff = total_msgs - 1
        new_to_summarize = messages[summarized_count:cutoff]
        recent_messages = messages[-1:]

        if new_to_summarize:
            history_text = "\n".join(
                [
                    f"### {'Developer' if isinstance(m, HumanMessage) else 'Assistant'} Turn:\n"
                    f"{sanitize_text_single_newline(m.content)[:1800]}"
                    for m in new_to_summarize
                ]
            )

            prompt_memoria = (
                "# PREVIOUS TECHNICAL SESSION MEMORY\n"
                f"{current_summary if current_summary else 'None (Start of session).'}\n"
                "# NEW DIALOGUE TURNS TO INTEGRATE\n"
                f"{history_text}\n"
                "# MEMORY SYNTHESIS INSTRUCTIONS (THIRD-PERSON TECHNICAL LOG)\n"
                "Write a comprehensive, accurate third-person technical memory summary that merges the previous "
                "memory with the new turns. Do NOT be vague—capture concrete technical details using these 4 bullets:\n"
                "- **Topic & User Goal:** What the developer is discussing and trying to build or solve.\n"
                "- **Developed Code & Sandbox Files:** Specific scripts/files created or read (e.g. `./sandbox/km_calculator_address.py`), functions implemented, and logic built so far.\n"
                "- **Chosen Stack, Libraries & Constraints:** Exact libraries/APIs chosen (e.g. `geopy`, `Nominatim`, `geodesic`), parameters agreed upon, and any forbidden packages or strict rules.\n"
                "- **Current State for Future Turns:** What was just completed and any pending context needed for follow-up questions."
            )

            try:
                res = helper_llm.invoke([HumanMessage(content=prompt_memoria)])
                current_summary = sanitize_text_single_newline(res.content)
                SESSION_MEMORY["summary"] = current_summary
                SESSION_MEMORY["summarized_count"] = cutoff
                print(f"[Memória LFM2.5-1.2B] Atualizada ({len(current_summary)} chars):\n{current_summary}")
            except Exception as exc:
                print(f"[Erro LFM2.5-1.2B]: {exc}")

        active_messages = recent_messages
    else:
        active_messages = messages

    prior_buffer = "\n".join(
        [
            f"- {'Developer' if isinstance(m, HumanMessage) else 'Assistant'}: "
            f"{sanitize_text_single_newline(m.content)[:600]}"
            for m in active_messages[:-1]
        ]
    )

    partes_memoria = list(filter(None, [current_summary, prior_buffer]))
    memoria_consolidada = "\n".join(partes_memoria) if partes_memoria else "Nenhum histórico."

    raw_user_input = active_messages[-1].content
    clean_user_input = sanitize_user_prompt(raw_user_input)

    return {
        "short_term_memory": memoria_consolidada,
        "clean_user_prompt": clean_user_input,
        "web_search_data": "Nenhuma busca web solicitada.",
    }


def tool_agent_search_node(state: AgentState):
    clean_prompt = state["clean_user_prompt"]
    CURRENT_TURN_CONTEXT["user_prompt"] = clean_prompt
    CURRENT_TURN_CONTEXT["memory_summary"] = state["short_term_memory"]
    CURRENT_TURN_CONTEXT["web_docs_fetched"] = ""
    TURN_EXECUTED_SIGNATURES.clear()

    existing_files = [p.name for p in sorted(SANDBOX_DIR.iterdir()) if p.is_file()]
    files_str = ", ".join(existing_files) if existing_files else "None"

    agent_input = (
        "Analyze the developer request below and execute the required tool(s). You may chain multiple tools if needed. /no_think\n"
        f"- **Files currently in ./sandbox/:** {files_str}\n"
        f"- **Prior Session Context:** {state['short_term_memory']}\n"
        "# MULTI-TOOL DECISION & CHAINING MATRIX\n"
        "1. **WEB SEARCH (`tavily_web_search`):**\n"
        "   - Call if you need to verify external library imports/methods (e.g. exact `geopy.distance` classes, recent FastAPI/PySpark APIs) or up-to-date documentation.\n"
        "   - After receiving the search result, evaluate if you also need to call `sandbox_file_manager`.\n"
        "2. **FILE WRITE (`sandbox_file_manager` with `operacao='write_file'`):**\n"
        "   - MUST CALL if the user asks to save, write, create, or store a file/script in the sandbox.\n"
        "   - If the user did not provide a filename, invent a descriptive snake_case filename.\n"
        "3. **FILE READ (`sandbox_file_manager` with `operacao='read_file'`):**\n"
        "   - MUST CALL if the user asks to read, open, inspect, or modify an existing file from `./sandbox/`.\n"
        "4. **STOPPING CONDITION:**\n"
        "   - If no tools are needed (or once all needed tools have returned their results), do NOT call any more tools and reply only `'DONE'`.\n"
        f"# DEVELOPER REQUEST\n{clean_prompt}"
    )

    web_text = "Nenhuma busca web solicitada."
    try:
        result = main_tool_agent.invoke(
            {"messages": [HumanMessage(content=agent_input)]},
            config={"recursion_limit": 6},
        )
        messages: List[BaseMessage] = result.get("messages", [])
        appended_tool_outputs: List[str] = []
        for msg in messages:
            if isinstance(msg, ToolMessage) and msg.content:
                content_str = str(msg.content)
                if (
                    "No relevant external web documentation found" not in content_str
                    and not content_str.startswith("ALREADY_")
                ):
                    tool_name = getattr(msg, "name", None) or "tool"
                    step_num = len(appended_tool_outputs) + 1
                    appended_tool_outputs.append(
                        f"## [TOOL #{step_num} RESULT — `{tool_name}`]\n{content_str}"
                    )
        if appended_tool_outputs:
            web_text = "\n".join(appended_tool_outputs)
    except Exception as exc:
        print(f"[Aviso Tool Agent]: {exc}")

    return {"web_search_data": web_text}


def prompt_builder_node(state: AgentState):
    final_messages = assemble_final_prompt(
        memoria=state["short_term_memory"],
        ide_prompt=state["system_prompt_ide"],
        dados_web=state["web_search_data"],
        prompt_usuario=state["clean_user_prompt"],
    )
    return {"final_prompt": final_messages}


workflow = StateGraph(AgentState)
workflow.add_node("memory_and_input_prep", memory_and_input_prep_node)
workflow.add_node("tool_agent_search", tool_agent_search_node)
workflow.add_node("prompt_builder", prompt_builder_node)

workflow.set_entry_point("memory_and_input_prep")
workflow.add_edge("memory_and_input_prep", "tool_agent_search")
workflow.add_edge("tool_agent_search", "prompt_builder")
workflow.add_edge("prompt_builder", END)
app_graph = workflow.compile()


# ==========================================
# 6. ENDPOINT SSE (VS CODE / CONTINUE.DEV)
# ==========================================
@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    lc_messages: List[BaseMessage] = []
    system_prompt_ide = ""

    for m in request.messages:
        text = normalize_content(m.content)
        if m.role == "system":
            system_prompt_ide += text + "\n"
        elif m.role == "user":
            lc_messages.append(HumanMessage(content=text))
        elif m.role == "assistant":
            lc_messages.append(AIMessage(content=text))

    if not lc_messages:
        raise HTTPException(status_code=400, detail="Sem mensagens do usuário.")

    initial_state: AgentState = {
        "messages": lc_messages,
        "system_prompt_ide": system_prompt_ide.strip(),
        "short_term_memory": "",
        "clean_user_prompt": "",
        "web_search_data": "",
        "final_prompt": [],
    }

    processed_state = app_graph.invoke(initial_state)
    final_prompt = processed_state["final_prompt"]

    req_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    created_time = int(time.time())

    if request.stream:
        async def sse_generator():
            try:
                async for chunk in main_llm.astream(final_prompt):
                    if chunk.content:
                        payload = {
                            "id": req_id,
                            "object": "chat.completion.chunk",
                            "created": created_time,
                            "model": request.model,
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": {"content": chunk.content},
                                    "finish_reason": None,
                                }
                            ],
                        }
                        yield f"data: {json.dumps(payload)}\n\n"

                end_payload = {
                    "id": req_id,
                    "object": "chat.completion.chunk",
                    "created": created_time,
                    "model": request.model,
                    "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                }
                yield f"data: {json.dumps(end_payload)}\n\n"
                yield "data: [DONE]\n\n"
            except Exception as stream_err:
                print(f"[Erro de Stream]: {stream_err}")
                yield "data: [DONE]\n\n"

        return StreamingResponse(sse_generator(), media_type="text/event-stream")

    response = await main_llm.ainvoke(final_prompt)
    return {
        "id": req_id,
        "object": "chat.completion",
        "created": created_time,
        "model": request.model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": response.content},
                "finish_reason": "stop",
            }
        ],
    }


if __name__ == "__main__":
    print(
        f"=== Iniciando LangGraph Proxy v0.5.0 (Helper: {HELPER_MODEL_NAME} | Main: {MAIN_MODEL_NAME}) na porta 8000 ==="
    )
    uvicorn.run(app, host="127.0.0.1", port=8000)