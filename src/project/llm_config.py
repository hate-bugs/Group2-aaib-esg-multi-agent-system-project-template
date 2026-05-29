import os
from dotenv import load_dotenv
from crewai import LLM

load_dotenv()


def _build_llm(*, enable_thinking: bool) -> LLM | None:
    model = os.getenv("LLM_MODEL_NAME")
    if not model:
        return None

    kwargs = {
        "model": model,
        "base_url": os.getenv("LLM_BASE_URL"),
        "api_key": os.getenv("LLM_API_KEY"),
        "temperature": 0.3,
        "max_tokens": 4096,
    }
    if not enable_thinking:
        kwargs["extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}}
    return LLM(**kwargs)


llm = _build_llm(enable_thinking=False)
llm_thinking = _build_llm(enable_thinking=True)
