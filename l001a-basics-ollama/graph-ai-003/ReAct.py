from typing import Annotated, Sequence, TypedDict
from langchain_core.messages import BaseMessage, ToolMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

# 1. Define your state
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

# 2. Define your tools
@tool
def add(a: int, b: int) -> int:
    """
    Tool to add two integers.

    Args:
        a (int): First integer.
        b (int): Second integer.

    Returns:
        int: The sum of the two integers.
    """
    print("Adding numbers...")
    return a + b

@tool
def multiply(a: int, b: int) -> int:
    """
    Tool to multiply two integers.

    Args:
        a (int): First integer.
        b (int): Second integer.

    Returns:
        int: The product of the two integers.
    """
    print("Multiplying numbers...")
    return a * b

@tool
def divide(a: int, b: int) -> float:
    """
    Tool to divide two integers.

    Args:
        a (int): First integer.
        b (int): Second integer.

    Returns:
        float: The result of the division.
    """
    print("Dividing numbers...")
    return a / b

@tool
def get_name() -> str:
    """
    Tool to get a random name.
    Returns:
        str: A random name
    """
    print("Getting a random name...")
    import random
    return random.choice(["Alice", "Bob", "Charlie", "David"])

tools = [add, multiply, divide, get_name]

# 3. Initialize the model
model = ChatOpenAI(
    api_key="ollama",
    model="llama3-groq-tool-use:70b",
    base_url="http://localhost:11434/v1",
).bind_tools(tools=tools, tool_choice="auto")

# 4. Node to call LLM, which may emit <tool_call>
def call_llm(state: AgentState) -> AgentState:
    system = SystemMessage(content="""
        You are a helpful AI that can call tools.
        Use tools when needed and return structured tool_call tags.
    """)
    response = model.invoke([system] + list(state["messages"]))
    return {"messages": [response]}

# 5. Inject tool execution using ToolNode
tool_node = ToolNode(tools)

# 6. Condition function
def should_continue(state: AgentState) -> str:
    last = state["messages"][-1]
    return "tools" if last.tool_calls else END

# 7. Build graph
graph = StateGraph(AgentState)
graph.add_node("llm", call_llm)
graph.add_node("tools", tool_node)
graph.set_entry_point("llm")
graph.add_conditional_edges("llm", should_continue, {"tools": "tools", END: END})
graph.add_edge("tools", "llm")
app = graph.compile()

# 8. Run with a prompt
def print_stream(stream):
    for s in stream:
        msg = s["messages"][-1]
        msg.pretty_print()

inputs = {"messages": [("user", 
    "Add 34 and 21 then subtract 10 from the result. Divide it by 2. Finally get the name and greet them with the final result."
)]}
print_stream(app.stream(inputs, stream_mode="values"))
