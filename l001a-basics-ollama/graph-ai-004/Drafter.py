from typing import Annotated, Sequence, TypedDict
from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, ToolMessage, SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

# Load .env for model config
load_dotenv()

# 1. Internal state for conversation
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

# Global document content
document_content = ""

# 2. Define tools
@tool
def update(content: str) -> str:
    """
    Tool to update the document content.
    """

    global document_content
    document_content = content
    return f"Document updated. Current content:\n{document_content}"

@tool
def save(filename: str) -> str:
    """
    Tool to save the document content to a file.
    """
    global document_content
    if not filename.endswith(".txt"):
        filename += ".txt"
    try:
        with open(filename, "w") as f:
            f.write(document_content)
        return f"Document saved to {filename}."
    except Exception as e:
        return f"Error saving document: {e}"

tools = [update, save]

# 3. Initialize local ChatOpenAI with tool binding
model = ChatOpenAI(
    api_key="ollama",
    model="llama3-groq-tool-use:70b",
    base_url="http://localhost:11434/v1",
).bind_tools(tools=tools, tool_choice="auto")

def our_agent(state: AgentState) -> AgentState:
    """
    This node will solve the user's query using the LLM.
    """
    
    system_prompt = SystemMessage(content=f"""
    You are Drafter, a helpful writing assistant. You are going to help the user update and modify documents.
    
    - If the user wants to update or modify content, use the 'update' tool with the complete updated content.
    - If the user wants to save and finish, you need to use the 'save' tool.
    - Make sure to always show the current document state after modifications.
    
    The current document content is:{document_content}
    """)


    response = model.invoke([system_prompt] + state['messages'])

    if not state['messages']:
        user_input = "I'm ready to help you update a document. What would you like to create?"
        user_message = HumanMessage(content=user_input)
    else:
        user_input = input("\n What would you like to do with the document? ")
        print(f"\nUser: {user_input}")
        user_message = HumanMessage(content=user_input)

    all_messages = [system_prompt] + list(state['messages']) + [user_message]

    response = model.invoke(all_messages)

    print(f"\nAI: {response.content}")
    if hasattr(response, 'tool_calls') and response.tool_calls:
        print(f"USING TOOLS: {[tc['name'] for tc in response.tool_calls]}")

    return {"messages": list(state['messages']) + [user_message, response]}

# 5. ToolNode to run actual tool calls
tool_node = ToolNode(tools=tools)

# 6. Continue until `save` runs
def should_continue(state: AgentState) -> str:
    for m in reversed(state["messages"]):
        if isinstance(m, ToolMessage) and m.name == "save":
            return END
    return "agent"

# 7. Build graph
graph = StateGraph(AgentState)
graph.add_node("agent", our_agent)
graph.add_node("tools", tool_node)
graph.set_entry_point("agent")
graph.add_edge("agent", "tools")
graph.add_edge("tools", "agent")
graph.add_conditional_edges("tools", should_continue, {"agent": "agent", END: END})
app = graph.compile()

# 8. Run loop
def print_messages(msgs):
    for m in msgs[-3:]:
        if isinstance(m, ToolMessage):
            print(f"[Tool] {m.name} → {m.content}")
        else:
            print(f"{m.content}")

def run():
    print("Welcome to Drafter!")
    state = {"messages": []}
    for step in app.stream(state, stream_mode="values"):
        if "messages" in step:
            print_messages(step["messages"])
    print("Goodbye!")

if __name__ == "__main__":
    run()
