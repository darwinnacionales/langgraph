import json

from flask import Flask, Response, request, stream_with_context
from flask_cors import CORS
from flask import render_template
from dotenv import load_dotenv

from .supervisor import workflow

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
    user_id = data.get("user_id", 0)
    if not user_input:
        return "Missing 'input'", 400

    def event_stream():
        yield sse("initial", f"Received your request: \"{user_input}\". Working on it…")
        print("SSE → initial:", user_input)

        for msg, metadata in workflow.stream(
            {"messages": [{"role": "user", "content": user_input}], "user_id": user_id},
            config={"configurable": {"thread_id": f"user-{user_id}", "checkpoint_ns": ""}},
            stream_mode="messages",
        ):
            node = metadata.get("langgraph_node", "unknown")
            content = getattr(msg, "content", "")
            msg_type = getattr(msg, "type", None)

            if msg_type == "tool":
                print(f"SSE → thought (tool): {node} → {content!r}")
                yield sse("thought", f"{node} ➜ {content}")

            elif node == "supervisor":
                # The content is a full Editor.js JSON object
                # We parse it, then stream each block individually
                try:
                    report_data = json.loads(content)
                    blocks = report_data.get("blocks", [])
                    print(f"SSE → Streaming {len(blocks)} report blocks.")
                    for block in blocks:
                        # Send each block as a 'report_block' event
                        yield sse("report_block", json.dumps(block))
                except json.JSONDecodeError:
                    # Fallback for plain text or malformed JSON
                    print(f"SSE → Could not parse report as JSON, sending as plain text.")
                    yield sse("report_block", json.dumps({
                        "type": "paragraph",
                        "data": { "text": content }
                    }))

        yield sse("final", "✅ Report complete.")
        print("SSE → final: ✅ Report complete.")

    return Response(stream_with_context(event_stream()), mimetype="text/event-stream")



def sse(event, data):
    """Correctly formats a string (potentially multi-line) into a single SSE message."""
    lines = data.split('\n')
    formatted_data = "\n".join(f"data: {line}" for line in lines)
    return f"event: {event}\n{formatted_data}\n\n"
