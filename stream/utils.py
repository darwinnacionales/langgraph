from typing import Sequence
from langgraph.graph.message import add_messages
from langchain_core.messages import AnyMessage
from langchain_core.messages.utils import trim_messages


def capped_add_messages(left: Sequence[AnyMessage], right: Sequence[AnyMessage]) -> Sequence[AnyMessage]:
    return add_messages(left, right)

def trim_history(state):
    trimmed = trim_messages(
        state["messages"],
        strategy="last",
        token_counter=len,
        max_tokens=70,
        start_on="human",
        end_on=("human", "tool"),
        include_system=True,
    )

    return {
        "llm_input_messages": trimmed,
    }
