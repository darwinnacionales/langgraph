import json
import random

from flask import Flask, Response, request, stream_with_context, render_template
from flask_cors import CORS
from dotenv import load_dotenv

from .workflow_graph import workflow_graph

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
    user_id = data.get("user_id")

    if not user_id:
        return "Missing 'user_id'", 400
    if not user_input:
        return "Missing 'input'", 400

    def event_stream():
        try:
            # Send an initial event (logged but no UI impact)
            yield sse("initial", "Session starting")

            sent_thoughts = set()
            report_started = False

            # Kick off the graph stream; RedisSaver ensures previous messages are loaded
            stream = workflow_graph.stream(
                {
                    "messages": [{"role": "user", "content": user_input}],
                    "user_id": user_id,
                    # "company": None,
                    # "time_duration": None,
                    # "report_type": None,
                },
                config={
                    "configurable": {
                        "thread_id": f"user-{user_id}",
                        "checkpoint_ns": ""
                    },
                    "recursion_limit": 100,
                },
                stream_mode=["messages", "custom"],
            )

            # Process each message event
            for event, data_tuple in stream:
                # print(f"[DEBUG] RAW EVENT → {event!r}")
                if event == "debug":
                    continue

                if event != "messages":
                    continue

                msg, _ = data_tuple
                content = getattr(msg, "content", "").strip()
                msg_type = getattr(msg, "type", None)

                if not content:
                    continue

                # Handle thought tool outputs
                if msg_type == 'ai' and hasattr(msg, 'tool_calls') and msg.tool_calls:
                    for tc in msg.tool_calls or []:
                        if tc.get('name') == 'notify_thought_tool':
                            args = json.loads(tc.get('args', '{}') or "{}")
                            ev = args.get("stage", "thought")
                            txt = args.get("thought", "")
                            if txt and txt not in sent_thoughts:
                                yield sse(ev, txt)
                                sent_thoughts.add(txt)
                    continue

                # Handle AI assistant messages
                if msg_type == 'ai':
                    # JSON report blocks
                    if content.startswith('{') and not report_started:
                        try:
                            lvl = json.loads(content)
                            if "blocks" in lvl:
                                report_started = True
                                for blk in lvl["blocks"]:
                                    yield sse("report_block", json.dumps(blk))
                        except json.JSONDecodeError:
                            pass
                    else:
                        yield sse("chat", content)
                    continue

                # Additional thought via tools
                if msg_type == 'tool' and getattr(msg, 'name', '') == "notify_thought_tool":
                    td = json.loads(content or "{}")
                    ev = td.get("event", "thought")
                    txt = td.get("content", "")
                    if txt and txt not in sent_thoughts:
                        yield sse(ev, txt)
                        sent_thoughts.add(txt)
                    continue

                # Ignore other messages
                continue

            # Send final event (logged, no UI effect)
            yield sse("final", "Session complete")

        except Exception as ex:
            print("Stream error:", ex)
            yield sse("error", f"Exception: {ex}")

    resp = Response(stream_with_context(event_stream()), mimetype="text/event-stream")
    resp.headers['X-Accel-Buffering'] = 'no'
    resp.headers['Cache-Control'] = 'no-cache'
    return resp

def sse(event: str, data: str) -> str:
    """Format server-sent event."""
    lines = str(data).split('\n')
    return f"event: {event}\n" + "\n".join(f"data: {l}" for l in lines) + "\n\n"
