import re
import string
import random

try:
    from math_verify import parse as _mv_parse, verify as _mv_verify
    _HAS_MV = True
except Exception:
    _HAS_MV = False

def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

def normalize_answer_math(s):

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):

        text = re.sub(r"(?<=\d)\.(?=\d)", "\x00", text)
        exclude = set(string.punctuation)
        text = "".join(ch for ch in text if ch not in exclude)
        return text.replace("\x00", ".")

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

def cal_f1_score(prediction, ground_truth, normalizer=normalize_answer):

    if isinstance(ground_truth, str):
        ground_truth_list = [ground_truth]
    else:
        ground_truth_list = ground_truth

    max_f1 = 0.0
    pred_tokens = normalizer(prediction).split()

    for gt in ground_truth_list:
        gold_tokens = normalizer(gt).split()
        common = set(pred_tokens) & set(gold_tokens)
        num_same = sum(min(pred_tokens.count(w), gold_tokens.count(w)) for w in common)

        if num_same == 0:
            f1 = 0.0
        else:
            precision = num_same / len(pred_tokens)
            recall = num_same / len(gold_tokens)
            f1 = 2 * precision * recall / (precision + recall)

        max_f1 = max(max_f1, f1)

    return max_f1

def exact_match_score(prediction, ground_truth, normalizer=normalize_answer):

    if isinstance(ground_truth, str):
        ground_truth_list = [ground_truth]
    else:
        ground_truth_list = ground_truth

    normalized_prediction = normalizer(prediction)

    for gt in ground_truth_list:
        if normalizer(gt) == normalized_prediction:
            return 1.0

    return 0.0

def _strict_norm_math(s):

    s = str(s)
    s = s.replace("\\left", "").replace("\\right", "")
    s = s.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac")
    for t in ("\\!", "\\,", "\\;", "\\ ", "\\quad", "\\qquad"):
        s = s.replace(t, "")
    s = s.replace("$", "")
    s = re.sub(r"\.0+(?=\D|$)", "", s)
    s = re.sub(r"\\(?:leqslant|leq|le)\b", "<=", s)
    s = re.sub(r"\\(?:geqslant|geq|ge)\b", ">=", s)
    s = re.sub(r"\\(?:neq|ne)\b", "!=", s)
    return re.sub(r"\s+", "", s).lower()

def _bracket_sig(s):

    return "".join(c for c in str(s) if c in "()[]")

def _neg_lead(s):

    return str(s).strip().lstrip("$ ").startswith("-")

def _pre_mv(s):

    return str(s).replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac")

def _strip_delim(s):

    s = str(s)
    for _d in ("\\(", "\\)", "\\[", "\\]"):
        s = s.replace(_d, "")
    return s.strip()

_SCI_RE = re.compile(r"\\times\s*10|\\cdot\s*10")
_CPLX_RE = re.compile(r"(?:\d|[+\-])\s*i(?![a-zA-Z])")

def _mv_unreliable(gt):

    g = gt.replace("\\left", "").replace("\\right", "")
    if any(op in g for op in ("=", "<", ">", "\\le", "\\ge", "\\ne")):
        return True
    return bool(_SCI_RE.search(gt) or _CPLX_RE.search(gt))

def math_equal(prediction, ground_truth):

    if prediction is None or ground_truth is None:
        return 0.0
    gts = [ground_truth] if isinstance(ground_truth, str) else list(ground_truth)
    pred = _strip_delim(prediction)
    for gt in gts:
        gt = _strip_delim(gt)
        if not _mv_unreliable(gt) and _HAS_MV:
            try:
                pg, ps = _mv_parse(_pre_mv(gt)), _mv_parse(_pre_mv(pred))
                if (pg and ps and _mv_verify(pg, ps)
                        and _bracket_sig(gt) == _bracket_sig(pred)
                        and _neg_lead(gt) == _neg_lead(pred)):
                    return 1.0
            except Exception:
                pass
        if _strict_norm_math(pred) == _strict_norm_math(gt):
            return 1.0
    return 0.0

def em_check(prediction, golden_answers):
    if isinstance(golden_answers, str):
        golden_answers = [golden_answers]
    normalized_prediction = normalize_answer(prediction)
    em_score = 0.0
    for golden_answer in golden_answers:
        golden_answer = normalize_answer(golden_answer)
        if golden_answer == normalized_prediction:
            em_score = 1.0
            break
    return em_score

def subem_check(prediction, golden_answers, normalizer=normalize_answer):
    if isinstance(golden_answers, str):
        golden_answers = [golden_answers]
    normalized_prediction = normalizer(prediction)
    sub_score = 0.0
    for golden_answer in golden_answers:
        golden_answer = normalizer(golden_answer)
        if golden_answer in normalized_prediction:
            sub_score = 1.0
            break
    return sub_score

def extract_solution(solution_str):

    answer_pattern = r'<answer>(.*?)</answer>'
    match = re.search(answer_pattern, solution_str, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None

def has_answer_tags(solution_str):

    if solution_str is None:
        return False
    answer_pattern = r'<answer>(.*?)</answer>'
    match = re.search(answer_pattern, solution_str, re.DOTALL)
    if match and match.group(1).strip():
        return True
    return False

def compute_score_format(solution_str):

    if solution_str is None:
        return 0.0
    try:
        assistant_blocks = re.findall(r'<\|im_start\|>assistant\n(.*?)<\|im_end\|>', solution_str, re.DOTALL)
        format_reward = 0.0
        if not assistant_blocks or len(assistant_blocks) == 0:
            return 0.0

        block = assistant_blocks[-1]
        m_think = re.search(r'<think>(.*?)</think>', block, re.DOTALL)
        m_answer = re.search(r'<answer>(.*?)</answer>', block, re.DOTALL)
        if m_think is not None:
            format_reward += 0.25
            if len(m_think.group(1).strip()) >= 30:
                format_reward += 0.25
        if m_answer is not None:
            format_reward += 0.25
            if len(m_answer.group(1).strip()) > 0:
                format_reward += 0.25
    except Exception as e:
        print(f"[DEBUG] Error in compute_score_format: {e}")
        return 0.0
    return format_reward

def compute_score_format_parts(solution_str):

    parts = {"fmt_think_tag": 0.0, "fmt_think_sub": 0.0, "fmt_ans_tag": 0.0, "fmt_ans_ne": 0.0}
    if solution_str is None:
        return parts
    try:
        blocks = re.findall(r'<\|im_start\|>assistant\n(.*?)<\|im_end\|>', solution_str, re.DOTALL)
        if not blocks:
            return parts
        block = blocks[-1]
        m_think = re.search(r'<think>(.*?)</think>', block, re.DOTALL)
        m_answer = re.search(r'<answer>(.*?)</answer>', block, re.DOTALL)
        if m_think is not None:
            parts["fmt_think_tag"] = 1.0
            if len(m_think.group(1).strip()) >= 30:
                parts["fmt_think_sub"] = 1.0
        if m_answer is not None:
            parts["fmt_ans_tag"] = 1.0
            if len(m_answer.group(1).strip()) > 0:
                parts["fmt_ans_ne"] = 1.0
    except Exception:
        pass
    return parts

def extract_think_text(solution_str):

    if solution_str is None:
        return ""
    try:
        blocks = re.findall(r'<\|im_start\|>assistant\n(.*?)<\|im_end\|>', solution_str, re.DOTALL)
        if not blocks:
            return ""
        m = re.search(r'<think>(.*?)</think>', blocks[-1], re.DOTALL)
        return m.group(1).strip() if m else ""
    except Exception:
        return ""

def compute_score_answer(solution_str, ground_truth, normalizer=normalize_answer):

    if solution_str is None:
        return 0.0
    try:
        assistant_blocks = re.findall(r'<\|im_start\|>assistant\n(.*?)<\|im_end\|>', solution_str, re.DOTALL)
        if not assistant_blocks or len(assistant_blocks) == 0:
            return 0.0
        solution_str = assistant_blocks[-1]

        if not has_answer_tags(solution_str):
            return 0.0
        answer = extract_solution(solution_str)
        answer_reward = 0.0
        if answer is not None:

            answer_reward = math_equal(answer, ground_truth)
    except Exception as e:
        print(f"[DEBUG] Error in compute_score_answer: {e}")
        return 0.0
    return answer_reward

def compute_score_sm(solution_str, ground_truth, normalizer=normalize_answer):

    if solution_str is None or ground_truth is None:
        return 0.0
    try:
        assistant_blocks = re.findall(r'<\|im_start\|>assistant\n(.*?)<\|im_end\|>', solution_str, re.DOTALL)
        if not assistant_blocks or len(assistant_blocks) == 0:
            return 0.0
        solution_str = assistant_blocks[-1]
        answer = extract_solution(solution_str)
        if answer is None:
            return 0.0
        sm_score = float(subem_check(answer, ground_truth, normalizer))
        return sm_score
    except Exception as e:
        print(f"[DEBUG] Error in compute_score_sm: {e}")
        return 0.0

def compute_score_em(solution_str, ground_truth, normalizer=normalize_answer):

    if solution_str is None or ground_truth is None:
        return 0.0
    try:
        assistant_blocks = re.findall(r'<\|im_start\|>assistant\n(.*?)<\|im_end\|>', solution_str, re.DOTALL)
        solution_str = assistant_blocks[-1]
        answer = extract_solution(solution_str)
        if answer is None:
            return 0.0

        em_score = math_equal(answer, ground_truth)
        return em_score
    except Exception as e:
        print(f"[DEBUG] Error in compute_score_em: {e}")
        return 0.0

def compute_score_f1(solution_str, ground_truth, normalizer=normalize_answer):

    if solution_str is None or ground_truth is None:
        return 0.0
    try:
        assistant_blocks = re.findall(r'<\|im_start\|>assistant\n(.*?)<\|im_end\|>', solution_str, re.DOTALL)
        solution_str = assistant_blocks[-1]
        answer = extract_solution(solution_str)
        if answer is None:
            return 0.0

        f1_score = cal_f1_score(answer, ground_truth, normalizer)
        return f1_score
    except Exception as e:
        print(f"[DEBUG] Error in compute_score_f1: {e}")
        return 0.0

def compute_score_format_answer(solution_str, ground_truth, normalizer=normalize_answer):

    if solution_str is None or ground_truth is None:
        return 0.0
    try:
        format_reward = compute_score_format(solution_str)
        answer_reward = compute_score_answer(solution_str, ground_truth, normalizer)
        sm_score_reward = compute_score_sm(solution_str, ground_truth, normalizer)
        format_reward = min(format_reward, 1.0)
        if format_reward == 1.0:
            rewards = -1.0 + format_reward + answer_reward
            return rewards
        else:
            rewards = -1.0 + format_reward
            return rewards
    except Exception as e:
        print(f"[DEBUG] Error in compute_score_format_answer: {e}")
        return -1.0
