import os

OPENAI_API_KEY = os.environ.get("VLLM_API_KEY", "EMPTY")
OPENAI_API_BASE = os.environ.get("VLLM_API_BASE", "")
MODEL_NAME = os.environ.get("VLLM_MODEL_NAME", "agent")
TEMPERATURE = float(os.environ.get("EVAL_TEMP", "0.7"))
TOP_P = float(os.environ.get("EVAL_TOP_P", "0.8"))
REPETITION_PENALTY = float(os.environ.get("EVAL_REPETITION_PENALTY", "1.05"))
