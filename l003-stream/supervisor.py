from typing import Annotated, Sequence, TypedDict
from langchain_core.messages import AnyMessage
from langgraph_supervisor import create_supervisor, create_handoff_tool
from langgraph.checkpoint.memory import InMemorySaver

from .utils import capped_add_messages, trim_history
from .models import gpt_41, gpt_41_mini
from .agents import data_agent, math_agent, verification_agent

class SupervisorState(TypedDict):
    messages: Annotated[Sequence[AnyMessage], capped_add_messages]
    remaining_steps: int
    user_id: int

supervisor = create_supervisor(
    agents=[
        data_agent,
        math_agent,
        verification_agent
    ],
    state_schema=SupervisorState,
    model=gpt_41_mini,
    pre_model_hook=trim_history,
    tools = [
        create_handoff_tool(agent_name="data_agent"),
        create_handoff_tool(agent_name="math_agent"),
        create_handoff_tool(agent_name="verification_agent")
    ],
    output_mode="full_history",
    prompt = """
    You are a report generation supervisor.
    Your task is to call the data_agent to gather data and then call math_agent to help you with the calculations.

    INSTRUCTIONS:
    1. You will receive a message from the user and you will respond by creating a report based on the request and the data you gather.
    2. You will call the data_agent to gather the necessary data.
    3. You will make an analysis of the data and then call the math_agent to perform calculations when necessary.
        - The math_agent can perform calculations such as min, max, average, and sum.
        - Therefore, you must call the math_agent after gathering the data.
        - DO NOT DO CALCULATIONS YOURSELF.
    4. You will then respond to the user with the report.
    5. Finally, you will have to call the verification_agent to verify the report.

    GUIDELINES:
    - Always respond in a professional and concise manner.
    - You must always call the data_agent first to gather the necessary data. DO NOT SKIP THIS STEP.
    - Do not make assumptions about the data. Always gather it first.
    - If you need to perform calculations, always call the math_agent after gathering the data.
    """
)

checkpointer = InMemorySaver()
workflow = supervisor.compile(
    checkpointer=checkpointer,
)
