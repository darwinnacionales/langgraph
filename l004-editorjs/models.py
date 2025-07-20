from langchain_openai import ChatOpenAI

gpt_41 = ChatOpenAI(
    model = "gpt-4.1",
    temperature = 0,
    max_tokens = None,
    timeout = None,
    max_retries = 3,
)

gpt_41_mini = ChatOpenAI(
    model = "gpt-4.1-mini",
    temperature = 0,
    max_tokens = None,
    timeout = None,
    max_retries = 3,
)
