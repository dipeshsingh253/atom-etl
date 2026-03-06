"""LangGraph agent workflow for question answering over ingested documents.

Integrates with LangSmith for full tracing when LANGCHAIN_TRACING_V2=true.
Every agent step is logged with rich thought-process visibility.
"""

import os
import time
import uuid
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
from src.modules.agent.citations import extract_citations
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

    if response.tool_calls:
        for tc in response.tool_calls:
            _log_thought(iteration, f"TOOL CALL: {tc['name']}")
            for k, v in tc["args"].items():
                arg_val = str(v)
                if len(arg_val) > 200:
                    arg_val = arg_val[:200] + "..."
                _log_thought(iteration, f"  └─ {k}: {arg_val}")

        if response.content and str(response.content).strip():
            reasoning = str(response.content).strip()
            _log_thought(iteration, f"REASONING: {reasoning[:500]}")
    else:
        answer_preview = str(response.content)[:300]
        _log_thought(iteration, "FINAL ANSWER GENERATED")
        _log_thought(iteration, f"  └─ Preview: {answer_preview}...")

    return {
        "messages": [response],
        "iteration_count": iteration,
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

        status = "✓ DATA FOUND" if has_data else "✗ NO DATA"
        _log_thought(iteration, f"TOOL RESULT: {msg.name} → {status} ({content_len} chars)")

        if has_data:
            lines = content.split("\n")
            preview_lines = lines[:6]
            for line in preview_lines:
                if line.strip():
                    _log_thought(iteration, f"  │ {line[:150]}")
            if len(lines) > 6:
                _log_thought(iteration, f"  │ ... ({len(lines) - 6} more lines)")
        else:
            _log_thought(iteration, f"  │ {content[:200]}")

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
    citations = await extract_citations(messages, answer)

    _log_thought(iteration, f"Answer length: {len(answer)} chars")
    _log_thought(iteration, f"Citations found: {len(citations)}")
    for c in citations:
        _log_thought(iteration, f"  └─ Page {c['page']}: {c['text'][:80]}...")

    return {
        "final_answer": answer,
        "citations": citations,
    }


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
    }

    # Generate a stable run ID so the caller can look up the trace in LangSmith.
    run_id = uuid.uuid4()
    _log_thought(0, f"Run ID: {run_id}")

    # Run the graph
    try:
        final_state = await app.ainvoke(initial_state, config={"run_id": run_id})
        answer = final_state.get("final_answer", "")
        citations = final_state.get("citations", [])
        total_iterations = final_state.get("iteration_count", 0)

        elapsed = time.time() - start_time

        _log_section("AGENT WORKFLOW COMPLETE")
        _log_thought(total_iterations, f"Total time: {elapsed:.1f}s")
        _log_thought(total_iterations, f"Iterations used: {total_iterations}")
        _log_thought(total_iterations, f"Answer length: {len(answer)} chars")
        _log_thought(total_iterations, f"Citations: {len(citations)} pages")
        _log_thought(total_iterations, f"Run ID: {run_id}")
        logger.info(_SEPARATOR)

        return {
            "answer": answer,
            "citations": citations,
            "langsmith_run_id": str(run_id),
        }

    except Exception as e:
        elapsed = time.time() - start_time
        _log_section("AGENT WORKFLOW FAILED")
        _log_thought(0, f"Error after {elapsed:.1f}s: {e}")
        logger.info(_SEPARATOR)

        return {
            "answer": f"An error occurred while processing your query: {str(e)}",
            "citations": [],
            "langsmith_run_id": str(run_id),
        }
