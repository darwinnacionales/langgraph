from flask import Flask, request, jsonify
from dotenv import load_dotenv
import os

from .supervisor import workflow

load_dotenv()

app = Flask(__name__)


@app.route('/chat', methods=['POST'])
def run_supervisor():
    try:
        data = request.get_json()
        user_input = data.get("input")
        user_id = data.get("user_id", 0)

        if not user_input:
            return jsonify({"error": "Missing 'input' field"}), 400

        # Run the supervisor
        config = {
            "configurable": {
                "thread_id": f"user-{user_id}"
            }
        }

        result = workflow.invoke(
            {
                "messages": [{"role": "user", "content": user_input}],
                "user_id": user_id,
            },
            config=config
        )

        print("\n=== ANSWER ===")
        print(result['messages'][-1].content)

        return jsonify({"response": result['messages'][-1].content})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
