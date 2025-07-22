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

If it does not work, check if the old redis is still running...
Try killing them..

```
ps aux | grep redis
```

Use kill -9 to kill the process

If using brew, run the following:
```
brew services stop redis
```
