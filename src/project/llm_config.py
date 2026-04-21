import os
from dotenv import load_dotenv
from crewai import LLM

load_dotenv()

# Qwen 3.5 llm with thinking disabled for short tasks
llm = LLM(
    model=os.getenv("LLM_MODEL_NAME"),
    base_url=os.getenv("LLM_BASE_URL"),
    api_key=os.getenv("LLM_API_KEY"),
    temperature=0.3,
    max_tokens=4096, #max token output of an agent
    extra_body={
        "chat_template_kwargs": {"enable_thinking": False} # disables the thinking process
    },
)

# Qwen 3.5 llm with thinking enabled for long or important tasks
llm_thinking = LLM(
    model=os.getenv("LLM_MODEL_NAME"),
    base_url=os.getenv("LLM_BASE_URL"),
    api_key=os.getenv("LLM_API_KEY"),
    temperature=0.3,
    max_tokens=4096, #max token output of an agent
)
