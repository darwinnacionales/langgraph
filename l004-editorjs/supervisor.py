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
        { "type": "header", "data": { "text": "Report Title", "level": 2 } },
        { "type": "paragraph", "data": { "text": "This is an introduction to the report." } },
        { "type": "list", "data": { "style": "unordered", "items": ["First point.", "Second point.", "Third point."] } },
        { "type": "paragraph", "data": { "text": "This is a concluding paragraph." } }
    ]
    }

    - **Do NOT** wrap the JSON in markdown.
    - Do not output any extra text — only **one valid JSON**.
    - Ensure the `time` is the current Unix time in milliseconds.
    - Use `header`, `paragraph`, and `list` blocks.
      - For headers, `level` can be 1, 2, 3, etc.
      - For lists, `style` can be "unordered" (for bullet points) or "ordered" (for numbers).

    Example:
    ```json
    {
    "time": 1752971214073,
    "blocks": [
        { "type": "header", "data": { "text": "Q1 Sales Analysis", "level": 2 } },
        { "type": "paragraph", "data": { "text": "The first quarter showed significant growth in key areas." } },
        { "type": "list", "data": { "style": "unordered", "items": [ "Total revenue increased by 15%.", "Customer acquisition grew by 22%.", "Product C was the top seller." ] } }
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
