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
    Your job:
    1. Call data_agent for data.
    2. Call math_agent for calculations.
    3. Compile a structured analysis.

    When the analysis is complete, output **exactly** a single JSON object formatted for Editor.js, like this:

    {
    "time": 1234567890123,
    "blocks": [
        { "type": "paragraph", "data": { "text": "Title here." } },
        { "type": "paragraph", "data": { "text": "Content here." } },
        { "type": "paragraph", "data": { "text": "Another content here." } },
    ]
    }

    - **Do NOT** wrap the JSON in markdown.
    - Do not output any extra text — only **one valid JSON**.
    - Ensure the `time` is the current Unix time in milliseconds.
    - Use only paragraphs as blocks.

    Example:
    ```json
    {
    "time": 1752971214073,
    "blocks": [
        { "type": "paragraph", "data": { "text": "Report Title" } },
        { "type": "paragraph", "data": { "text": "This is the content of the report." } },
        { "type": "paragraph", "data": { "text": "Additional information can go here." } }
    ]
    }
    ```

    Always produce exactly one JSON object, and the supervisor-furnished content should be the JSON.
    """
)

checkpointer = InMemorySaver()
workflow = supervisor.compile(
    checkpointer=checkpointer,
)
