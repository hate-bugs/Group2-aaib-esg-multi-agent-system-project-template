import os
from dotenv import load_dotenv
from crewai import LLM

load_dotenv()

# Environment-configurable LLM settings (all have sensible defaults):
# LLM_MODEL_NAME: model identifier (e.g. "qwen-3.5"), used by crewai.LLM
# LLM_BASE_URL: base URL for the LLM API endpoint
# LLM_API_KEY: API key for authentication with the LLM provider
# LLM_TEMPERATURE: sampling temperature (float) controlling randomness; default 0.3
# LLM_MAX_TOKENS: maximum tokens allowed in model responses; default 4096
# LLM_ENABLE_THINKING: whether to enable the model's internal 'thinking' feature

LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME")
LLM_BASE_URL = os.getenv("LLM_BASE_URL")
LLM_API_KEY = os.getenv("LLM_API_KEY")

# Sampling temperature for generated responses (0.0 deterministic, higher => more random)
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.3"))

# Maximum tokens the assistant may return. Keep this below your model/account limits.
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "4096"))

# Enable the model's internal 'thinking' process (if supported by the backend). The
# value is interpreted as a boolean-like string ("1","true","yes" -> True).
_think_env = os.getenv("LLM_ENABLE_THINKING", "false").lower()
LLM_ENABLE_THINKING = _think_env in ("1", "true", "yes")


# Basic runtime validation: fail early with a clear error if required
# environment variables are missing. This prevents passing `None` into the
# LLM constructor which can lead to confusing errors later at runtime.
if not LLM_MODEL_NAME:
    raise RuntimeError("Environment variable LLM_MODEL_NAME must be set (e.g. 'qwen-3.5')")

if not LLM_API_KEY:
    raise RuntimeError("Environment variable LLM_API_KEY must be set with your provider API key")


# Qwen 3.5 LLM instance tuned for short tasks (thinking optionally disabled)
llm = LLM(
    model=LLM_MODEL_NAME,
    base_url=LLM_BASE_URL,
    api_key=LLM_API_KEY,
    temperature=LLM_TEMPERATURE,
    max_tokens=LLM_MAX_TOKENS,  # max token output of an agent
    extra_body={
        # chat_template_kwargs.enable_thinking toggles the provider's thinking feature
        "chat_template_kwargs": {"enable_thinking": LLM_ENABLE_THINKING}
    },
)


# Qwen 3.5 LLM instance intended for longer/important tasks (thinking controlled
# by the same environment flag). We reuse the same env-driven parameters so both
# instances remain consistent and configurable without changing code.
llm_thinking = LLM(
    model=LLM_MODEL_NAME,
    base_url=LLM_BASE_URL,
    api_key=LLM_API_KEY,
    temperature=LLM_TEMPERATURE,
    max_tokens=LLM_MAX_TOKENS,  # max token output of an agent
)
