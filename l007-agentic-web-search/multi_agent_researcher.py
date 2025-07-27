import re

from langchain_ollama import ChatOllama
from langgraph.prebuilt import create_react_agent
from langchain_community.tools import DuckDuckGoSearchRun
from langgraph_supervisor import create_supervisor

def clean_text(text: str):
    cleaned_text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    return cleaned_text.strip()

model = ChatOllama(model="qwen3:8b")

query_refiner_prompt = (
    "You are a query refiner agent.\n\n"
    "INSTRUCTIONS:\n"
    "- Refine the user's query to make it more specific and actionable.\n"
    "- Respond ONLY with the refined query, do NOT include any other text or comments.\n"
)

research_prompt = (
    "You are a research agent.\n\n"
    "INSTRUCTIONS:\n"
    "- Assist ONLY with research-related tasks. DO NOT do anything else.\n"
    "- After you're done with your research, respond to the supervisor directly\n"
    "- Respond ONLY with the summary of the results, do NOT include any other text or comments.\n"
)

supervisor_prompt = (
    "You are a supervisor agent.\n\n"
    "INSTRUCTIONS:\n"
    "- Manage the workflow between the query refiner and the research agent.\n"
    "- Use the Query Refiner Agent to correct typos and enhance query for better results.\n"
    "- ALWAYS use Research Agent to retrieve up-to-date information from the web after query refinement.\n"
    "- You MUST NOT provide answers from your own knowledge. ALWAYS transfer to research_agent for web search.\n"
    "- Follow this exact workflow: 1) Transfer to Query Refiner → 2) Transfer to Research Agent → 3) Provide final summary\n"
    "- Respond ONLY with the final summary of the results, do NOT include any other text or comments.\n"
)

query_refiner_agent = create_react_agent(
    model=model,
    tools=[],
    prompt=query_refiner_prompt,
    name="query_refiner_agent"
)

research_agent = create_react_agent(
    model=model,
    tools=[DuckDuckGoSearchRun()],
    prompt=research_prompt,
    name="research_agent"
)

supervisor_agent = create_supervisor(
    model=model,
    agents=[query_refiner_agent, research_agent],
    prompt=supervisor_prompt,
)

app = supervisor_agent.compile()



result = app.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": "What was the latest news about Trump's tariffs on the Filipins?"
            }
        ]
    }
)

for message in result["messages"]:
    print(message.pretty_print())
    print("-" * 100)
