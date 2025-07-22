import json

from typing import Annotated, Sequence, TypedDict, Optional
from langchain_core.messages import AnyMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.redis import RedisSaver

from .supervisor import supervisor
from .models import gpt_41_mini

# --- State schema ---
class MiniAgentState(TypedDict):
    messages: Annotated[Sequence[AnyMessage], ...]
    user_id: int
    company: Optional[str]
    time_duration: Optional[str]
    report_type: Optional[str]

# --- LLM-based extraction ---
EXTRACTION_SYSTEM_PROMPT = """
Extract the following information from the user conversation, if possible:
- Company (the company for the report)
- Time duration (the time period to cover)
- Report type (the kind of report, e.g., financial, summary, comparison, etc.)

If you cannot find an answer for a field, return null for it.

Reply in this exact JSON format:

{
  "company": null,
  "time_duration": null,
  "report_type": null
}
"""

def extract_context_llm(messages):
    chat = [{"role": "system", "content": EXTRACTION_SYSTEM_PROMPT}]
    print("-> [extract_context_llm] messages:")
    for m in messages:
        if isinstance(m, dict):
            role = m.get("role", "user")
            content = m.get("content", "")
        else:
            role = getattr(m, "role", None)
            content = getattr(m, "content", "") or ""
            if role is None:
                role = "assistant" if isinstance(m, AIMessage) else "user"

        print(f"   • role='{role}' content='{content}'")
        chat.append({"role": role, "content": content})

    try:
        result = gpt_41_mini.invoke(chat)
        data = json.loads(result.content)
        print(f"-> [extract_context_llm] extracted: {data!r}")
        return data.get("company"), data.get("time_duration"), data.get("report_type")
    except Exception as e:
        print("-> [extract_context_llm] LLM error:", e)
        return None, None, None



# --- Debug node ---
def debug_node(state: MiniAgentState) -> MiniAgentState:
    print("=== Restored conversation history ===")
    for msg in state["messages"]:
        print(f"{getattr(msg, 'role', '?')}: {getattr(msg, 'content', '')}")
    print("Current context:", state["company"], state["time_duration"], state["report_type"])
    return state

# --- get_context node ---
def get_context_node(state: MiniAgentState) -> MiniAgentState:
    msgs = state["messages"]
    company = state.get("company")
    time_duration = state.get("time_duration")
    report_type = state.get("report_type")

    # try LLM extraction if missing
    if not (company and time_duration and report_type):
        nc, nt, nr = extract_context_llm(msgs)
        company = company or nc
        time_duration = time_duration or nt
        report_type = report_type or nr

    # ask for missing field, but don't lose other fields
    if not company:
        return {
            "messages": msgs + [AIMessage(content="Which company is this report for?")],
            "user_id": state["user_id"],
            "company": None,
            "time_duration": time_duration,
            "report_type": report_type,
        }
    if not time_duration:
        return {
            "messages": msgs + [AIMessage(content="What time duration should I cover in the report?")],
            "user_id": state["user_id"],
            "company": company,
            "time_duration": None,
            "report_type": report_type,
        }
    if not report_type:
        return {
            "messages": msgs + [AIMessage(content="What type of report do you need (e.g., financial, summary)?")],
            "user_id": state["user_id"],
            "company": company,
            "time_duration": time_duration,
            "report_type": None,
        }

    # all set
    return {
        "messages": msgs,
        "user_id": state["user_id"],
        "company": company,
        "time_duration": time_duration,
        "report_type": report_type,
    }

# --- supervisor node ---
def supervisor_node(state: MiniAgentState) -> MiniAgentState:
    enriched = list(state["messages"]) + [
        HumanMessage(
            content=(
                f"Company: {state['company']}, "
                f"Time duration: {state['time_duration']}, "
                f"Report type: {state['report_type']}"
            ),
            role="user"
        )
    ]
    result = supervisor.invoke({
        "messages": enriched,
        "user_id": state["user_id"]
    })
    return {
        "messages": result["messages"],
        "user_id": state["user_id"],
        "company": state["company"],
        "time_duration": state["time_duration"],
        "report_type": state["report_type"],
    }

# --- Build graph ---
graph = StateGraph(MiniAgentState)
graph.add_node("get_context", get_context_node)
graph.add_node("debug", debug_node)
graph.add_node("supervisor", supervisor_node)

# Branching logic
graph.add_conditional_edges("get_context", {
    "get_context": lambda s: not (s["company"] and s["time_duration"] and s["report_type"]),
    "debug":       lambda s:     (s["company"] and s["time_duration"] and s["report_type"])
})
graph.add_edge("debug", "supervisor")
graph.add_edge("supervisor", END)
graph.set_entry_point("get_context")

# --- RedisSaver ---
with RedisSaver.from_conn_string("redis://localhost:6379") as checkpointer:
    checkpointer.setup()
    workflow_graph = graph.compile(checkpointer=checkpointer)
