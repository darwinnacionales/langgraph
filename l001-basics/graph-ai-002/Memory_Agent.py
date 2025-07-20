import os
from typing import TypedDict, List, Union
from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from dotenv import load_dotenv

load_dotenv()

class AgentState(TypedDict):
    messages: List[Union[HumanMessage, AIMessage]]

llm = ChatOpenAI(model="gpt-4.1-mini")

def process(state: AgentState) -> AgentState:
    """
    This node will solve the user's query using the LLM.
    """
    response = llm.invoke(state['messages'])

    # Append the AI response to the messages
    state['messages'].append(AIMessage(content=response.content))
    
    return state

graph = StateGraph(AgentState)
graph.add_node("process", process)
graph.add_edge(START, "process")
graph.add_edge("process", END)

agent = graph.compile()

conversation_history = []

user_input = input("Enter: ")
while user_input != "exit":
    conversation_history.append(HumanMessage(content=user_input))

    result = agent.invoke({
        "messages": conversation_history
    })

    print(f"\nAI: {result['messages'][-1].content}")
    conversation_history.append(result['messages'][-1])

    user_input = input("Enter: ")

with open("conversation_history.txt", "w") as f:
    f.write("Conversation History:\n")

    for message in conversation_history:
        if isinstance(message, HumanMessage):
            f.write(f"User: {message.content}\n")
        elif isinstance(message, AIMessage):
            f.write(f"AI: {message.content}\n")

    f.write("\nEnd of conversation history.\n")

print("Conversation history saved to conversation_history.txt")