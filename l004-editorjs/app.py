import json
import random

from flask import Flask, Response, request, stream_with_context, render_template
from flask_cors import CORS
from dotenv import load_dotenv

from .supervisor import workflow

import sys
import os

if not sys.stdout.isatty():
    os.environ['PYTHONUNBUFFERED'] = "1"

load_dotenv()
app = Flask(__name__, template_folder="templates")
CORS(app)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat_stream():
    data = request.get_json() or {}
    user_input = data.get("input")
    user_id = random.randint(1, 100000)

    if not user_input:
        return "Missing 'input'", 400

    def event_stream():
        try:
            print("SSE → user input:", user_input)

            sent_thoughts = set()
            report_started = False

            stream = workflow.stream(
                {"messages": [{"role": "user", "content": user_input}], "user_id": user_id},
                config={
                    "configurable": {
                        "thread_id": f"user-{user_id}",
                        "checkpoint_ns": ""
                    },
                    "recursion_limit": 100,
                },
                stream_mode=["messages", "custom", "debug"],
            )

            for event, data_tuple in stream:
                print("-" * 80)
                print(f"RAW EVENT: {repr(event)}")
                print(f"RAW DATA: {repr(data_tuple)}")
                print("-" * 80)

                if event == "debug":
                    # We only care about when the specific thought tool finishes running
                    if (data.get("event") == "on_tool_end" and 
                        data.get("name") == "notify_thought_tool"):
                        try:
                            # The tool's direct JSON output is in the 'data' payload
                            tool_output_str = data.get("data", {}).get("output")
                            if tool_output_str:
                                tool_data = json.loads(tool_output_str)
                                event_name = tool_data.get("event", "thought")
                                text = tool_data.get("content", "")

                                # Ensure we haven't sent this thought already
                                if text and text not in sent_thoughts:
                                    print(f"SSE (from DEBUG) → {event_name}: {text}")
                                    yield sse(event_name, text)
                                    sent_thoughts.add(text)
                        except Exception as e:
                            print(f"SSE → (error parsing DEBUG event): {e}")
                    # After processing a debug event, we skip to the next item
                    continue

                if event == "custom":
                    print(f"SSE → custom event: {data_tuple}")
                    continue

                if event != "messages":
                    continue

                if not (isinstance(data_tuple, tuple) and len(data_tuple) == 2):
                    print(f"Warning: Unexpected data structure for 'messages' event: {data_tuple}")
                    continue

                msg, metadata = data_tuple
                content = getattr(msg, "content", "")
                msg_type = getattr(msg, "type", None)

                # --- Priority 1: Thought tool call inside AIMessage ---
                if msg_type == 'ai' and hasattr(msg, 'tool_calls') and msg.tool_calls:
                    for tool_call in msg.tool_calls:
                        if tool_call.get('name') == 'notify_thought_tool':
                            try:
                                raw_args = tool_call.get('args', '{}')
                                args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
                                event_name = args.get("stage", "thought")
                                text = args.get("thought", "")
                                if text and text not in sent_thoughts:
                                    print(f"SSE (from AIMessage tool_call) → {event_name}: {text}")
                                    yield sse(event_name, text)
                                    sent_thoughts.add(text)
                            except Exception as e:
                                print(f"[ERROR] Failed to parse tool_call args: {e}")
                        else:
                            print(f"SSE → (ignored non-thought tool_call): {tool_call.get('name', 'unknown')}")

                # --- Priority 2: Final report JSON in plain AIMessage ---
                elif msg_type == 'ai' and not (hasattr(msg, 'tool_calls') and msg.tool_calls):
                    if isinstance(content, str) and content.strip().startswith('{') and not report_started:
                        try:
                            report_data = json.loads(content)
                            if "blocks" in report_data:
                                print(f"SSE → Streaming {len(report_data['blocks'])} report blocks.")
                                report_started = True
                                for block in report_data.get("blocks", []):
                                    yield sse("report_block", json.dumps(block))
                            else:
                                print(f"SSE → (ignored intermediate JSON): {content}")
                        except json.JSONDecodeError:
                            print(f"SSE → (ignored non-JSON or malformed content): {content[:200]}")

                # --- Priority 3: Fallback ToolMessage (outside AI tool_call) ---
                elif msg_type == "tool" and getattr(msg, 'name', '') == "notify_thought_tool":
                    try:
                        tool_data = json.loads(content or "{}")
                        event_name = tool_data.get("event", "thought")
                        text = tool_data.get("content", "")
                        if text and text not in sent_thoughts:
                            print(f"SSE (from ToolMessage) → {event_name}: {text}")
                            yield sse(event_name, text)
                            sent_thoughts.add(text)
                        else:
                            print(f"SSE → (ignored duplicate ToolMessage): {text}")
                    except (json.JSONDecodeError, AttributeError, TypeError) as e:
                        print(f"SSE → (error parsing ToolMessage content): {e}")

                # --- Priority 4: Debug unknown types ---
                else:
                    tool_name = getattr(msg, 'name', '')
                    node = metadata.get("langgraph_node", "unknown")
                    print(f"SSE → (ignored event) Type: {msg_type}, Node: {node}, Tool: {tool_name}")

        except Exception as e:
            print(f"[ERROR] Exception in event stream: {e}")
            yield sse("error", f"An error occurred: {str(e)}")

    response = Response(stream_with_context(event_stream()), mimetype="text/event-stream")
    response.headers['X-Accel-Buffering'] = 'no'
    response.headers['Cache-Control'] = 'no-cache'
    return response

def sse(event, data):
    """Format a string into a valid SSE message."""
    lines = str(data).split('\n')
    formatted_data = "\n".join(f"data: {line}" for line in lines)
    return f"event: {event}\n{formatted_data}\n\n"
