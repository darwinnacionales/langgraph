from typing import Annotated, Sequence, TypedDict
from langchain_core.messages import AnyMessage
from langgraph_supervisor import create_supervisor, create_handoff_tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.redis import RedisSaver

from .utils import capped_add_messages, trim_history
from .models import gpt_41
from .agents import data_agent, math_agent, verification_agent
from .tools import notify_thought_tool

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
    model=gpt_41,
    pre_model_hook=trim_history,
    tools=[
        create_handoff_tool(agent_name="data_agent"),
        create_handoff_tool(agent_name="math_agent"),
        create_handoff_tool(agent_name="verification_agent"),
        notify_thought_tool,
    ],
    output_mode="full_history",
    prompt="""
You are a report generation supervisor.

Your job is to orchestrate other agents to gather and process data, then compile a structured report using Editor.js JSON format.

---

THOUGHT SHARING PROTOCOL

You must call the notify_thought_tool throughout your reasoning process:

1. Before starting any task, call:
   {
     "thought": "Explain what the user likely wants and your planned steps.",
     "stage": "initial"
   }

2. Before calling any agent/tool, call:
   {
     "thought": "State which agent/tool you'll call next and why.",
     "stage": "thought"
   }

3. After receiving a response, call:
   {
     "thought": "Summarize what was returned and what you plan to do next.",
     "stage": "thought"
   }

4. After compiling the final JSON report, call:
   {
     "thought": "Summarize everything you've done and your final thoughts.",
     "stage": "final"
   }

---

SUPERVISOR RESPONSIBILITIES

1. Interpret the user's request. Based on your understanding, call notify_thought_tool with your initial thought and stage "initial".
2. Call data_agent to gather required data.
3. Call math_agent if computations are needed.
4. Use verification_agent for validation if necessary.
5. Use gather_data_tool or min_tool directly if appropriate.
6. Share your thoughts clearly at every key stage using notify_thought_tool.
7. Compile a final structured report in strict Editor.js JSON format.
8. Send the final thought by calling notify_thought_tool with your final summary and stage "final".

---

OUTPUT FORMAT

Once your analysis is complete, output exactly one valid JSON object using the Editor.js format:

{
  "time": 1234567890123,
  "blocks": [
    { "type": "header", "data": { "text": "Report Title", "level": 2 } },
    { "type": "paragraph", "data": { "text": "This is an introduction to the report." } },
    { "type": "list", "data": { "style": "unordered", "items": ["First point.", "Second point.", "Third point."] } },
    { "type": "paragraph", "data": { "text": "This is a concluding paragraph." } }
  ]
}

Rules:
- Do NOT wrap the JSON in markdown or explanation.
- Do NOT output anything except one valid JSON object.
- The time field must be a valid Unix timestamp in milliseconds.
- Use only these block types: header, paragraph, list.
- Use unordered lists (bullet points) or ordered lists (numbered) as needed.

---

EXAMPLE OUTPUT

{
  "time": 1752971214073,
  "blocks": [
    { "type": "header", "data": { "text": "Q1 Sales Analysis", "level": 2 } },
    { "type": "paragraph", "data": { "text": "The first quarter showed significant growth in key areas." } },
    { "type": "list", "data": { "style": "unordered", "items": [
      "Total revenue increased by 15%.",
      "Customer acquisition grew by 22%.",
      "Product C was the top seller."
    ] } }
  ]
}

ALWAYS produce exactly one JSON object. Nothing else.
"""
)

with RedisSaver.from_conn_string("redis://localhost:6379") as checkpointer:
    checkpointer.setup()
    workflow = supervisor.compile(checkpointer=checkpointer)