from flask import Flask, Response, request, stream_with_context
from flask_cors import CORS
from flask import render_template
from dotenv import load_dotenv
import os
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
                print(f"SSE → report chunk: {content!r}")
                # split and send each line separately
                for line in content.splitlines():
                    yield sse("report", line)

        yield sse("final", "✅ Report complete.")
        print("SSE → final: ✅ Report complete.")

    return Response(stream_with_context(event_stream()), mimetype="text/event-stream")



def sse(event, data):
    return f"event: {event}\ndata: {data}\n\n"
