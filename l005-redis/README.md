# Dependencies

Run:
pip install -r requirements.txt

.env
```bash
OPENAI_API_KEY=your_openai_api_key
```

# Run

flask run


# Redis
For LangGraph to work with Redis, we need redis-stack-server installed and running.
You can install it via Homebrew on macOS:
```bash
brew tap redis-stack/redis-stack

brew install --cask redis-stack
```

Then, start the Redis server:
```bash
redis-stack-server
```