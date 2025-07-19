from typing import Annotated, Sequence, TypedDict
from dotenv import load_dotenv  
from langchain_core.messages import BaseMessage # The foundational class for all message types in LangGraph
from langchain_core.messages import ToolMessage # Passes data back to LLM after it calls a tool such as the content and the tool_call_id
from langchain_core.messages import SystemMessage # Message for providing instructions to the LLM
from langchain_core.messages import HumanMessage # Represents a message from the user
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

load_dotenv()

document_content = ""

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

@tool
def update(content: str) -> str:
    """
    Updates the document content with the provided text.
    """
    global document_content
    document_content = content
    return "Document has been updated successfully. The current content is:\n" + document_content

@tool
def save(filename: str) -> str:
    """
    Saves the current document content to a file.

    Args:
        filename (str): The name of the text file to save the document content to.
    """
    
    if not filename.endswith('.txt'):
        filename += '.txt'

    global document_content

    try:
        with open(filename, 'w') as file:
            file.write(document_content)
        print(f"Document content saved to {filename}.")
        return f"Document has been saved successfully to {filename}."
    except Exception as e:
        return f"An error occurred while saving the document: {str(e)}"

tools = [update, save]

model = ChatOpenAI(model="gpt-4.1-mini").bind_tools(tools)

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

def should_continue(state: AgentState) -> str:
    """
    This node will determine if the conversation should continue.
    """
    messages = state['messages']

    if not messages:
        return "continue"
    
    # This looks for the most recent tool messsage
    for message in reversed(messages):
        # and checks if this is a ToolMessage resulting from save
        if isinstance(message, ToolMessage) and message.name == "save":
            return "end"
        
    return "continue"

def print_messages(messages):
    if not messages:
        return
    
    for message in messages[-3:]:
        if isinstance(message, ToolMessage):
            print(f"Tool Call: {message.name} - {message.content}")

graph = StateGraph(AgentState)
graph.add_node("agent", our_agent)
graph.add_node("tools", ToolNode(tools=tools))

graph.set_entry_point("agent")

graph.add_edge("agent", "tools")

graph.add_conditional_edges(
    "tools",
    should_continue,
    {
        "continue": "agent",
        "end": END
    }
)

app = graph.compile()

def run_document_agent():
    print("Welcome to Drafter, your document drafting assistant!")

    state = {"messages": []}

    for step in app.stream(state, stream_mode="values"):
        if "messages" in step:
            print_messages(step["messages"])

    print("\nThank you for using Drafter! Your document has been processed.")

if __name__ == "__main__":
    run_document_agent()