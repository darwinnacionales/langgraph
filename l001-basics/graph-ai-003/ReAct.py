from typing import Annotated, Sequence, TypedDict
from dotenv import load_dotenv  
from langchain_core.messages import BaseMessage # The foundational class for all message types in LangGraph
from langchain_core.messages import ToolMessage # Passes data back to LLM after it calls a tool such as the content and the tool_call_id
from langchain_core.messages import SystemMessage # Message for providing instructions to the LLM
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode


load_dotenv()

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

@tool
def add(a: int, b: int) -> int:
    """
    Adds two numbers together.
    """
    return a + b

@tool
def multiply(a: int, b: int) -> int:
    """
    Multiplies two numbers together.
    """
    return a * b

@tool
def divide(a: int, b: int) -> int:
    """
    Divide two numbers together.
    """
    return a / b

tools = [add, multiply, divide]

model = ChatOpenAI(model="gpt-4.1-mini").bind_tools(tools)

def model_call(state: AgentState) -> AgentState:
    """
    This node will solve the user's query using the LLM.
    """

    system_prompt = SystemMessage(
        content="You are a helpful AI assistant. You will answer the user's query to the best of your ability. If you need to use a tool, you will call it and return the result to the user."
    )

    response = model.invoke([system_prompt] + state['messages'])

    return { "messages": [response] }


def should_continue(state: AgentState) -> str:
    """
    This node will determine if the conversation should continue.
    """
    last_message = state['messages'][-1]
    
    if not last_message.tool_calls:
        return "end"
    else:
        return "continue"
    
graph = StateGraph(AgentState)
graph.add_node("our_agent", model_call)

tool_node = ToolNode(tools=tools)
graph.add_node("tools", tool_node)

graph.set_entry_point("our_agent")

graph.add_conditional_edges(
    "our_agent",
    should_continue,
    {
        "continue": "tools",
        "end": END
    }
)

graph.add_edge("tools", "our_agent")

app = graph.compile()

def print_stream(stream):
    for s in stream:
        message = s["messages"][-1]
        if isinstance(message, tuple):
            print(message)
        else:
            message.pretty_print()

inputs = {"messages": [("user", "Add 34 and 21 then subtract 10 from the result. Finally, divide the result by 2.")]}
print_stream(app.stream(inputs, stream_mode="values"))