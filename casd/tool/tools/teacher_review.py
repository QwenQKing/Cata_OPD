from typing import List, Dict, Optional, Union
import os
import re
import asyncio
import unicodedata

import httpx
from openai import AsyncOpenAI
try:
    from openai import RateLimitError, APITimeoutError, APIConnectionError, InternalServerError
    _RETRYABLE = (RateLimitError, APITimeoutError, APIConnectionError, InternalServerError)
except Exception:
    _RETRYABLE = (Exception,)

for _k in ("http_proxy", "https_proxy", "all_proxy", "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY"):
    os.environ.pop(_k, None)

TEACHER_API_KEY = os.environ.get("TEACHER_API_KEY", "").strip()

TEACHER_BASE_URL = os.environ.get("TEACHER_BASE_URL", "").strip()
TEACHER_MODEL = os.environ.get("TEACHER_MODEL", "").strip()

TEACHER_TEMPERATURE = float(os.environ.get("TEACHER_TEMPERATURE", "0.3"))
TEACHER_MAX_TOKENS = int(os.environ.get("TEACHER_MAX_TOKENS", "1024"))
TEACHER_TIMEOUT = float(os.environ.get("TEACHER_TIMEOUT", "120"))
_TEACHER_MAX_CONCURRENCY = int(os.environ.get("TEACHER_MAX_CONCURRENCY", "8"))
_MAX_RETRY_NET = 3
_BACKOFF_BASE = 1.5

_loop_state: Dict = {}

def _get_loop_state():

    loop = asyncio.get_running_loop()
    for _dead in [l for l in _loop_state if l.is_closed()]:
        _loop_state.pop(_dead, None)
    st = _loop_state.get(loop)
    if st is None:
        missing = [name for name, value in (
            ("TEACHER_BASE_URL", TEACHER_BASE_URL),
            ("TEACHER_MODEL", TEACHER_MODEL),
            ("TEACHER_API_KEY", TEACHER_API_KEY),
        ) if not value]
        if missing:
            raise RuntimeError("Missing teacher configuration: " + ", ".join(missing))
        client = AsyncOpenAI(
            api_key=TEACHER_API_KEY,
            base_url=TEACHER_BASE_URL,
            http_client=httpx.AsyncClient(trust_env=False, timeout=TEACHER_TIMEOUT),
        )
        st = (client, asyncio.Semaphore(_TEACHER_MAX_CONCURRENCY))
        _loop_state[loop] = st
    return st

TEACHER_SYSTEM = (
    "You are a teacher. You are given a Problem and its reference answer (FOR YOUR EYES "
    "ONLY — you must NEVER give the answer, or any part of it, to the student). The "
    "student will show you their thinking and answer, formatted as <think> reasoning "
    "</think><answer> final answer </answer>.\n\n"
    "Problem:\n{question}\n\n"
    "Reference answer (FOR YOUR EYES ONLY — never reveal): {ground_truth}\n\n"
    "Based on the student's most recent response, give the correct step-by-step METHOD for "
    "solving this problem, but do NOT give the answer. Write a COMPLETE numbered method (use as "
    "many steps as the problem needs):\n"
    "    Step 1: ...\n    Step 2: ...\n    ...\n"
    "Each step describes WHAT to do, so the student can re-derive the answer and solve "
    "similar problems on their own.\n\n"
    "Hard constraints — you must NEVER:\n"
    "- reveal the reference answer or any value taken from it;\n"
    "- give anything that lets the student copy the answer instead of deriving it."
)
_USER = "Student's most recent response (their reasoning and answer):\n{attempt}"

def _gt_to_list(gt: Optional[Union[str, List[str]]]) -> List[str]:
    if gt is None:
        return []
    return [str(x) for x in gt] if isinstance(gt, (list, tuple)) else [str(gt)]

def _normalize(s: str) -> str:

    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = re.sub(r"[^0-9a-z一-鿿 ]+", " ", s.lower())
    return re.sub(r"\s+", " ", s).strip()

def _leaks(guidance: str, gt_list: List[str]) -> bool:

    g = _normalize(guidance)
    g = re.sub(r"(?<![0-9a-z])step\s+\d+", " ", g)
    for a in gt_list:
        na = _normalize(a)
        if not na:
            continue
        if re.search(r"(?<![0-9a-z])" + re.escape(na) + r"(?![0-9a-z])", g):
            return True
    return False

def verify(student_output: str, ground_truth: Union[str, List[str]],
           data_source: str = "hotpotqa/hotpot_qa") -> bool:

    if not student_output:
        return False
    solution_str = f"<|im_start|>assistant\n{student_output}<|im_end|>"
    try:
        from casd.src.reward_score import _default_compute_score_answer
        return float(_default_compute_score_answer(data_source, solution_str, ground_truth)) >= 1.0
    except Exception as e:
        print(f"[teacher_review] verify failed: {e}")
        return False

class TeacherReviewer:

    def __init__(self, question: str, ground_truth: Optional[Union[str, List[str]]] = None):
        self.question = question
        self.gt_list = _gt_to_list(ground_truth)
        gt_str = " / ".join(self.gt_list) if self.gt_list else "(not provided)"

        self.messages: List[Dict[str, str]] = [{
            "role": "system",
            "content": TEACHER_SYSTEM.format(question=question, ground_truth=gt_str)}]

        self.n_calls = 0
        self.in_tokens = 0
        self.out_tokens = 0
        self.n_ok = 0
        self.n_fail = 0

    async def _create_once(self):

        client, sem = _get_loop_state()
        last_exc = None
        for _att in range(_MAX_RETRY_NET + 1):
            try:
                async with sem:
                    self.n_calls += 1
                    return await client.chat.completions.create(
                        model=TEACHER_MODEL, messages=self.messages,
                        temperature=TEACHER_TEMPERATURE, max_tokens=TEACHER_MAX_TOKENS)
            except _RETRYABLE as e:
                last_exc = e
                if _att < _MAX_RETRY_NET:
                    await asyncio.sleep(_BACKOFF_BASE * (2 ** _att))
        raise last_exc

    async def guide_async(self, student_attempt: str) -> Optional[str]:

        self.messages.append({"role": "user", "content": _USER.format(attempt=student_attempt)})
        try:
            resp = await self._create_once()
            _usage = getattr(resp, "usage", None)
            if _usage is not None:
                self.in_tokens += int(getattr(_usage, "prompt_tokens", 0) or 0)
                self.out_tokens += int(getattr(_usage, "completion_tokens", 0) or 0)
            guidance = (resp.choices[0].message.content or "").strip()
            if not guidance:

                self.n_fail += 1
                return None
            self.messages.append({"role": "assistant", "content": guidance})
            self.n_ok += 1
            return guidance
        except Exception as e:
            self.n_fail += 1
            print(f"[teacher_review] guide failed: {e}")
            return None

    def guide(self, student_attempt: str) -> Optional[str]:
        return asyncio.run(self.guide_async(student_attempt))

def get_teacher_hint(question: str, student_attempt: str,
                     ground_truth: Optional[Union[str, List[str]]] = None) -> Optional[str]:

    return TeacherReviewer(question, ground_truth).guide(student_attempt)
