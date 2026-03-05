"""LangGraph agent workflow for question answering over ingested documents.

Integrates with LangSmith for full tracing when LANGCHAIN_TRACING_V2=true.
Every agent step is logged with rich thought-process visibility.
"""

import json
import os
import re
import time
from typing import Annotated, Any, Optional, TypedDict

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from pydantic import SecretStr
from langgraph.prebuilt import ToolNode
from loguru import logger
from sqlalchemy.ext.asyncio import AsyncSession

from src.core.config import get_settings
from src.modules.agent.prompts import SYSTEM_PROMPT
from src.modules.agent.tools.math_tool import (
    calculate_arithmetic,
    calculate_cagr,
    calculate_percentage,
    calculate_percentage_change,
)
from src.modules.agent.tools.retrieval import retrieve_documents
from src.modules.agent.tools.sql_tool import (
    SQL_SCHEMA_DESCRIPTION,
    run_sql_query,
    set_db_session,
)

# Maximum agent iterations before forcing termination
MAX_AGENT_ITERATIONS = 15

# ── Logging helpers ──────────────────────────────────────────────────────────

_SEPARATOR = "─" * 72


def _log_section(title: str) -> None:
    """Print a visible section header in logs."""
    logger.info("")
    logger.info(_SEPARATOR)
    logger.info(f"  {title}")
    logger.info(_SEPARATOR)


def _log_thought(iteration: int, message: str) -> None:
    """Log an agent thought with iteration prefix for easy grep."""
    logger.info(f"  [Step {iteration}] {message}")


# ── LangSmith configuration ─────────────────────────────────────────────────

def _configure_langsmith() -> None:
    """Set LangSmith environment variables from application settings.

    Env vars read by the langsmith/langchain SDK:
      LANGSMITH_TRACING, LANGSMITH_API_KEY, LANGSMITH_PROJECT, LANGSMITH_ENDPOINT
    """
    settings = get_settings()

    api_key = settings.langsmith_api_key
    project = settings.langsmith_project
    endpoint = settings.langsmith_endpoint
    tracing = settings.langsmith_tracing

    has_real_key = (
        api_key
        and api_key != "your-langsmith-api-key-here"
    )

    if tracing and has_real_key:
        os.environ["LANGSMITH_TRACING"] = "true"
        os.environ["LANGSMITH_API_KEY"] = api_key
        os.environ["LANGSMITH_PROJECT"] = project
        os.environ["LANGSMITH_ENDPOINT"] = endpoint
        logger.info(f"LangSmith tracing ENABLED → project: {project}")
    else:
        os.environ["LANGSMITH_TRACING"] = "false"
        os.environ.pop("LANGSMITH_API_KEY", None)
        if not has_real_key:
            logger.debug(
                "LangSmith tracing DISABLED (no valid API key configured)"
            )


# ── Agent state ──────────────────────────────────────────────────────────────

class AgentState(TypedDict):
    """State maintained throughout the agent workflow."""

    messages: Annotated[list, add_messages]
    query: str
    document_id: Optional[str]
    citations: list[dict]
    final_answer: str
    iteration_count: int
    execution_trace: list[dict]


# All available tools
TOOLS = [
    retrieve_documents,
    run_sql_query,
    calculate_cagr,
    calculate_percentage,
    calculate_percentage_change,
    calculate_arithmetic,
]


def _get_llm():
    """Get the configured LLM for the agent."""
    settings = get_settings()
    return ChatOpenAI(
        model=settings.openai_model,
        api_key=SecretStr(settings.openai_api_key) if settings.openai_api_key else None,
        temperature=0,
    ).bind_tools(TOOLS)


# ── Graph nodes ──────────────────────────────────────────────────────────────

def _should_continue(state: AgentState) -> str:
    """Determine if the agent should continue calling tools or generate final answer."""
    messages = state["messages"]
    last_message = messages[-1]
    iteration_count = state.get("iteration_count", 0)

    if iteration_count >= MAX_AGENT_ITERATIONS:
        logger.warning(
            f"  [Router] Hit iteration limit ({MAX_AGENT_ITERATIONS}) → forcing END"
        )
        return "end"

    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        tool_names = [tc["name"] for tc in last_message.tool_calls]
        _log_thought(
            iteration_count,
            f"DECISION: Need more data → calling tools: {tool_names}",
        )
        return "tools"

    _log_thought(iteration_count, "DECISION: Have enough data → generating final answer")
    return "end"


async def agent_node(state: AgentState) -> dict[str, Any]:
    """The main agent reasoning node — decides which tools to call.

    This is where the LLM "thinks". We log:
      - What iteration we're on
      - Which tools the LLM chose and why (args reveal strategy)
      - Or if it decided it has enough info to answer
    """
    llm = _get_llm()
    iteration = state.get("iteration_count", 0) + 1

    _log_section(f"AGENT REASONING — Iteration {iteration}")

    # Show what context the agent has so far
    tool_results_so_far = [
        m for m in state["messages"] if isinstance(m, ToolMessage)
    ]
    if tool_results_so_far:
        _log_thought(iteration, f"Context: {len(tool_results_so_far)} tool results available from previous steps")
    else:
        _log_thought(iteration, "Context: First interaction — no tool results yet")

    response = await llm.ainvoke(state["messages"])

    # Build trace entry
    trace_entry: dict[str, Any] = {
        "step": iteration,
        "node": "agent",
        "timestamp": time.time(),
    }

    if response.tool_calls:
        trace_entry["action"] = "tool_calls"
        trace_entry["tools"] = []

        for tc in response.tool_calls:
            tool_info = {
                "name": tc["name"],
                "args": {k: str(v)[:300] for k, v in tc["args"].items()},
            }
            trace_entry["tools"].append(tool_info)

            # Rich logging — show exactly what the agent is doing and why
            _log_thought(iteration, f"TOOL CALL: {tc['name']}")
            for k, v in tc["args"].items():
                arg_val = str(v)
                if len(arg_val) > 200:
                    arg_val = arg_val[:200] + "..."
                _log_thought(iteration, f"  └─ {k}: {arg_val}")

        # Log the agent's reasoning text if it included any
        if response.content and str(response.content).strip():
            reasoning = str(response.content).strip()
            _log_thought(iteration, f"REASONING: {reasoning[:500]}")
            trace_entry["reasoning"] = reasoning[:500]
    else:
        trace_entry["action"] = "final_response"
        answer_preview = str(response.content)[:300]
        _log_thought(iteration, "FINAL ANSWER GENERATED")
        _log_thought(iteration, f"  └─ Preview: {answer_preview}...")
        trace_entry["response_preview"] = answer_preview

    trace = state.get("execution_trace", []) + [trace_entry]

    return {
        "messages": [response],
        "iteration_count": iteration,
        "execution_trace": trace,
    }


async def tool_node_with_trace(state: AgentState) -> dict[str, Any]:
    """Execute tools and record results in the execution trace.

    Logs what each tool returned so we can see the data flowing
    back to the agent.
    """
    iteration = state.get("iteration_count", 0)
    _log_section(f"TOOL EXECUTION — Iteration {iteration}")

    tool_node = ToolNode(TOOLS)
    result = await tool_node.ainvoke(state)

    trace_entries = []
    for msg in result.get("messages", []):
        if not isinstance(msg, ToolMessage):
            continue

        content = str(msg.content) if msg.content else ""
        content_len = len(content)
        has_data = (
            "no results" not in content.lower()
            and "error" not in content.lower()
            and content_len > 20
        )

        trace_entries.append({
            "step": iteration,
            "node": "tool_result",
            "tool_name": msg.name,
            "timestamp": time.time(),
            "result_length": content_len,
            "result_preview": content[:500],
            "has_data": has_data,
        })

        # Rich logging of tool results
        status = "✓ DATA FOUND" if has_data else "✗ NO DATA"
        _log_thought(iteration, f"TOOL RESULT: {msg.name} → {status} ({content_len} chars)")

        # Show a meaningful preview of what the tool returned
        if has_data:
            # For SQL results, show first few rows
            lines = content.split("\n")
            preview_lines = lines[:6]
            for line in preview_lines:
                if line.strip():
                    _log_thought(iteration, f"  │ {line[:150]}")
            if len(lines) > 6:
                _log_thought(iteration, f"  │ ... ({len(lines) - 6} more lines)")
        else:
            _log_thought(iteration, f"  │ {content[:200]}")

    trace = state.get("execution_trace", []) + trace_entries
    result["execution_trace"] = trace
    return result


async def generate_final_answer(state: AgentState) -> dict[str, Any]:
    """Extract the final answer and citations from the agent's conversation."""
    iteration = state.get("iteration_count", 0)
    _log_section("GENERATING FINAL ANSWER")

    messages = state["messages"]

    # Find the last AI message (the final answer)
    last_ai_message = None
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and not msg.tool_calls:
            last_ai_message = msg
            break

    answer = last_ai_message.content if last_ai_message else "I could not generate an answer."

    # Extract citations
    citations = _extract_citations(messages, answer)

    _log_thought(iteration, f"Answer length: {len(answer)} chars")
    _log_thought(iteration, f"Citations found: {len(citations)}")
    for c in citations:
        _log_thought(iteration, f"  └─ Page {c['page']}: {c['text'][:80]}...")

    # Final trace entry
    trace = state.get("execution_trace", []) + [{
        "step": iteration + 1,
        "node": "generate_final_answer",
        "timestamp": time.time(),
        "answer_length": len(answer),
        "citations_count": len(citations),
        "cited_pages": [c["page"] for c in citations],
    }]

    return {
        "final_answer": answer,
        "citations": citations,
        "execution_trace": trace,
    }


# ── Citation helpers ──────────────────────────────────────────────────────────


def _is_discovery_query(sql_query: str, result_content: str) -> bool:
    """Detect bulk discovery queries that list all tables/visuals."""
    sql_upper = sql_query.upper()
    # Queries selecting directly from metadata tables without joining data tables
    if re.search(r'FROM\s+document_tables\b', sql_upper, re.IGNORECASE) and 'table_rows' not in sql_query.lower():
        return True
    if re.search(r'FROM\s+document_visuals\b', sql_upper, re.IGNORECASE) and 'visual_data' not in sql_query.lower():
        return True
    # Many page_number mentions = bulk listing
    if result_content.count("'page_number'") > 15:
        return True
    return False


def _extract_table_name_from_context(sql_query: str, result_content: str) -> str | None:
    """Extract the table name from SQL WHERE clause or query results."""
    # From WHERE clause: table_name LIKE '%...%' or table_name = '...'
    m = re.search(r"table_name\s+(?:LIKE|=|ILIKE)\s+'%?([^%']+)%?'", sql_query, re.IGNORECASE)
    if m:
        return m.group(1).strip()
    # From visual title
    m = re.search(r"title\s+(?:LIKE|=|ILIKE)\s+'%?([^%']+)%?'", sql_query, re.IGNORECASE)
    if m:
        return m.group(1).strip()
    # From result data — first table_name value
    m = re.search(r"'table_name':\s*'([^']+)'", result_content)
    if m:
        return m.group(1).strip()
    return None


def _truncate_at_sentence(text: str, max_len: int = 250) -> str:
    """Truncate text at the nearest sentence boundary within max_len."""
    if len(text) <= max_len:
        return text
    truncated = text[:max_len]
    for sep in ['. ', '.\n', '? ', '! ']:
        last_idx = truncated.rfind(sep)
        if last_idx > max_len // 3:
            return truncated[: last_idx + 1].strip()
    last_space = truncated.rfind(' ')
    if last_space > max_len // 3:
        return truncated[:last_space].strip() + "..."
    return truncated.strip() + "..."


def _parse_retrieval_evidence(content: str, evidence: list[dict]) -> None:
    """Parse vector search results into citation evidence with real text passages."""
    blocks = content.split("\n\n---\n\n")
    for block in blocks:
        header_match = re.match(
            r'\[Result \d+\]\s*Page\s+(\d+)(?:\s*\([^)]*\))*\s*:\s*\n?(.*)',
            block,
            re.DOTALL,
        )
        if not header_match:
            continue
        page = int(header_match.group(1))
        text_content = header_match.group(2).strip()
        if not text_content or len(text_content) < 10:
            continue
        passage = _truncate_at_sentence(text_content, 250)
        evidence.append({
            "page": page,
            "text": passage,
            "source": "document_text",
        })


def _build_table_page_map(messages: list) -> dict[str, int]:
    """Build a table_name → page_number map from SQL results (including discovery queries).

    This captures metadata from bulk discovery queries that we don't cite directly
    but whose page info is useful for targeted queries that lack page_number.
    """
    table_pages: dict[str, int] = {}
    for msg in messages:
        if not isinstance(msg, ToolMessage) or msg.name != "run_sql_query":
            continue
        content = str(msg.content) if msg.content else ""
        # Match rows like {'table_name': 'X', ... 'page_number': 'Y'}
        for m in re.finditer(
            r"'table_name':\s*'([^']+)'[^}]*'page_number':\s*'(\d+)'", content
        ):
            table_pages[m.group(1).lower()] = int(m.group(2))
        # Also reversed order
        for m in re.finditer(
            r"'page_number':\s*'(\d+)'[^}]*'table_name':\s*'([^']+)'", content
        ):
            table_pages[m.group(2).lower()] = int(m.group(1))
    return table_pages


def _parse_sql_evidence(
    sql_query: str,
    content: str,
    evidence: list[dict],
    table_page_map: dict[str, int] | None = None,
) -> None:
    """Parse SQL results into citation evidence with table name and page."""
    if not content or "error" in content.lower()[:50] or "no results" in content.lower()[:30]:
        return
    if _is_discovery_query(sql_query, content):
        return

    table_name = _extract_table_name_from_context(sql_query, content)

    # Collect unique pages from results
    pages: set[int] = set()
    for m in re.finditer(r"'page_number':\s*'(\d+)'", content):
        pages.add(int(m.group(1)))

    # If no pages in SELECT results, try the table→page map from discovery queries
    if not pages and table_name and table_page_map:
        tn_lower = table_name.lower()
        for key, page in table_page_map.items():
            if tn_lower in key or key in tn_lower:
                pages.add(page)
                break

    if not pages and not table_name:
        return

    # Count rows
    row_match = re.search(r'Rows returned:\s*(\d+)', content)
    row_count = int(row_match.group(1)) if row_match else 0

    # Build descriptive text that a user can verify
    if table_name:
        desc = f"Table: {table_name}"
        if row_count:
            desc += f" ({row_count} data rows)"
    else:
        desc = f"SQL query result ({row_count} rows)" if row_count else "SQL query result"

    # Add a sample data row for verifiability
    data_rows = [l for l in content.split('\n') if l.startswith('{')]
    if data_rows:
        sample = data_rows[0]
        if len(sample) > 200:
            sample = sample[:200] + "..."
        desc += f"\nSample: {sample}"

    # Create one citation per unique page, capped at 3 pages per SQL query
    if pages:
        for page in sorted(pages)[:3]:
            evidence.append({
                "page": page,
                "text": desc,
                "source": "table",
                "table_name": table_name,
            })
    elif table_name:
        # Table found but page unknown — still include it
        evidence.append({
            "page": 0,
            "text": desc,
            "source": "table",
            "table_name": table_name,
        })


def _parse_math_evidence(
    tool_name: str, tool_call_info: dict, content: str, evidence: list[dict]
) -> None:
    """Parse math tool results into citation evidence."""
    if "error" in content.lower()[:50]:
        return
    args = tool_call_info.get("args", {})

    if tool_name == "calculate_cagr":
        desc = f"CAGR calculation: start={args.get('start_value')}, end={args.get('end_value')}, periods={args.get('periods')}"
    elif tool_name == "calculate_percentage":
        desc = f"Percentage: {args.get('part')} of {args.get('whole')}"
    elif tool_name == "calculate_percentage_change":
        desc = f"Change: from {args.get('old_value')} to {args.get('new_value')}"
    elif tool_name == "calculate_arithmetic":
        desc = f"Arithmetic: {args.get('expression', 'calculation')}"
    else:
        desc = f"Calculation: {tool_name}"

    desc += f"\nResult: {content.strip()[:200]}"
    evidence.append({"page": 0, "text": desc, "source": "calculation"})


def _build_final_citations(
    evidence: list[dict], answer_pages: set[int]
) -> list[dict]:
    """Build deduplicated citation list, prioritising pages mentioned in the answer."""
    if not evidence:
        return []

    seen_keys: set[str] = set()
    prioritized: list[dict] = []
    secondary: list[dict] = []

    for ev in evidence:
        page = ev["page"]
        source = ev["source"]
        key = f"{page}:{source}:{ev.get('table_name', '')}"
        if key in seen_keys:
            continue
        seen_keys.add(key)

        if page in answer_pages:
            prioritized.append(ev)
        else:
            secondary.append(ev)

    prioritized.sort(key=lambda x: x["page"])
    secondary.sort(key=lambda x: x["page"])

    combined = prioritized + secondary
    final: list[dict] = []
    for c in combined:
        if c["page"] == 0 and c["source"] in ("calculation", "table"):
            # Keep calculations and tables even without a known page
            final.append(c)
        elif c["page"] > 0:
            final.append(c)
    return final[:5]


def _extract_citations(messages: list, final_answer: str) -> list[dict]:
    """Extract meaningful, verifiable citations from tool evidence.

    For text passages: provides actual searchable text from the page.
    For tables: references the table name and page number.
    For calculations: references the formula and result.
    """
    evidence: list[dict] = []

    # Pages mentioned in the final answer (for prioritization)
    answer_pages: set[int] = set()
    for m in re.finditer(r'[Pp]age\s+(\d+)', final_answer):
        answer_pages.add(int(m.group(1)))

    # Map tool_call_id → tool call info (so we can get SQL query text / math args)
    tool_call_map: dict[str, Any] = {}
    for msg in messages:
        if isinstance(msg, AIMessage) and msg.tool_calls:
            for tc in msg.tool_calls:
                tc_id = tc.get("id") if isinstance(tc, dict) else getattr(tc, "id", None)
                if tc_id:
                    tool_call_map[tc_id] = dict(tc) if not isinstance(tc, dict) else tc

    # Build table→page map from discovery queries for page resolution
    table_page_map = _build_table_page_map(messages)

    # Walk every ToolMessage and extract evidence
    math_tools = {
        "calculate_cagr", "calculate_percentage",
        "calculate_percentage_change", "calculate_arithmetic",
    }
    for msg in messages:
        if not isinstance(msg, ToolMessage):
            continue
        content = str(msg.content) if msg.content else ""
        tc_info = tool_call_map.get(msg.tool_call_id, {})

        if msg.name == "retrieve_documents":
            _parse_retrieval_evidence(content, evidence)
        elif msg.name == "run_sql_query":
            sql_query = tc_info.get("args", {}).get("sql_query", "")
            _parse_sql_evidence(sql_query, content, evidence, table_page_map)
        elif msg.name in math_tools:
            _parse_math_evidence(msg.name, tc_info, content, evidence)

    return _build_final_citations(evidence, answer_pages)


def build_graph() -> StateGraph:
    """Build the LangGraph agent workflow.

    Workflow:
        agent → (has tool calls?) → tools → agent → ... → end → generate_final_answer
    """
    graph = StateGraph(AgentState)

    # Add nodes
    graph.add_node("agent", agent_node)
    graph.add_node("tools", tool_node_with_trace)
    graph.add_node("generate_final_answer", generate_final_answer)

    # Set entry point
    graph.set_entry_point("agent")

    # Add conditional edges
    graph.add_conditional_edges(
        "agent",
        _should_continue,
        {
            "tools": "tools",
            "end": "generate_final_answer",
        },
    )

    # Tools always return to agent
    graph.add_edge("tools", "agent")

    # Final answer ends the graph
    graph.add_edge("generate_final_answer", END)

    return graph


# TODO: No need to return the execution trace to the client in the final API response if we log it fully in our backend and using langsmith as well. 
async def run_agent(
    query: str,
    db: AsyncSession,
    document_id: Optional[str] = None,
) -> dict:
    """Run the agent workflow to answer a query.

    Args:
        query: The user's question.
        db: Async database session for SQL queries.
        document_id: Optional document ID to scope the search.

    Returns:
        Dict with 'answer', 'citations', and 'execution_trace'.
    """
    start_time = time.time()

    # Configure LangSmith tracing (sets env vars that LangGraph reads)
    # TODO: We can do this during app startup instead of on every query if we want to optimize slightly.
    _configure_langsmith()

    # Set DB session for SQL tool
    set_db_session(db)

    _log_section("AGENT WORKFLOW START")
    _log_thought(0, f"Query: {query}")
    if document_id:
        _log_thought(0, f"Document scope: {document_id}")
    _log_thought(0, f"Max iterations: {MAX_AGENT_ITERATIONS}")
    _log_thought(0, f"Available tools: {[t.name for t in TOOLS]}")

    settings = get_settings()
    if settings.langsmith_tracing and settings.langsmith_api_key and settings.langsmith_api_key != "your-langsmith-api-key-here":
        _log_thought(0, f"LangSmith: ENABLED → project '{settings.langsmith_project}'")
    else:
        _log_thought(0, "LangSmith: DISABLED (no API key or tracing off)")

    # Build and compile the graph
    graph = build_graph()
    app = graph.compile()

    # Prepare initial state
    system_msg = SystemMessage(content=SYSTEM_PROMPT + "\n\n" + SQL_SCHEMA_DESCRIPTION)

    query_context = query
    if document_id:
        query_context = f"[Document ID: {document_id}] {query}"

    initial_state: AgentState = {
        "messages": [system_msg, HumanMessage(content=query_context)],
        "query": query,
        "document_id": document_id,
        "citations": [],
        "final_answer": "",
        "iteration_count": 0,
        "execution_trace": [{
            "step": 0,
            "node": "start",
            "timestamp": start_time,
            "query": query,
            "document_id": document_id,
        }],
    }

    # Run the graph
    try:
        final_state = await app.ainvoke(initial_state)
        answer = final_state.get("final_answer", "")
        citations = final_state.get("citations", [])
        execution_trace = final_state.get("execution_trace", [])
        total_iterations = final_state.get("iteration_count", 0)

        elapsed = time.time() - start_time

        # Completion trace entry
        execution_trace.append({
            "step": len(execution_trace),
            "node": "complete",
            "timestamp": time.time(),
            "total_time_seconds": round(elapsed, 2),
            "total_iterations": total_iterations,
            "answer_length": len(answer),
            "citations_count": len(citations),
        })

        _log_section("AGENT WORKFLOW COMPLETE")
        _log_thought(total_iterations, f"Total time: {elapsed:.1f}s")
        _log_thought(total_iterations, f"Iterations used: {total_iterations}")
        _log_thought(total_iterations, f"Answer length: {len(answer)} chars")
        _log_thought(total_iterations, f"Citations: {len(citations)} pages")
        _log_thought(total_iterations, f"Trace steps: {len(execution_trace)}")
        logger.info(_SEPARATOR)

        return {
            "answer": answer,
            "citations": citations,
            "execution_trace": execution_trace,
        }

    except Exception as e:
        elapsed = time.time() - start_time
        _log_section("AGENT WORKFLOW FAILED")
        _log_thought(0, f"Error after {elapsed:.1f}s: {e}")
        logger.info(_SEPARATOR)

        return {
            "answer": f"An error occurred while processing your query: {str(e)}",
            "citations": [],
            "execution_trace": [{
                "step": 0,
                "node": "error",
                "timestamp": time.time(),
                "error": str(e),
                "total_time_seconds": round(elapsed, 2),
            }],
        }
