_MATH_SOURCES = {"gsm8k", "mawps", "gsm_plus", "svamp"}

def _is_math(data_source):

    s = str(data_source).lower()
    return "math" in s or s in _MATH_SOURCES

def _normalizer(data_source):

    from . import qa_em_and_format
    return (qa_em_and_format.normalize_answer_math if _is_math(data_source)
            else qa_em_and_format.normalize_answer)

def _to_float(res):
    return float(res) if isinstance(res, (int, float, bool)) else float(res[0])

def _default_compute_score_format(data_source, solution_str, extra_info=None):

    from . import qa_em_and_format
    return _to_float(qa_em_and_format.compute_score_format(solution_str))

def _default_compute_score_answer(data_source, solution_str, ground_truth, extra_info=None):
    from . import qa_em_and_format
    return _to_float(qa_em_and_format.compute_score_em(
        solution_str, ground_truth, normalizer=_normalizer(data_source)))

def _default_compute_score_sm(data_source, solution_str, ground_truth, extra_info=None):
    from . import qa_em_and_format
    return _to_float(qa_em_and_format.compute_score_sm(
        solution_str, ground_truth, normalizer=_normalizer(data_source)))

def _default_compute_score_f1(data_source, solution_str, ground_truth, extra_info=None):
    from . import qa_em_and_format
    return _to_float(qa_em_and_format.compute_score_f1(
        solution_str, ground_truth, normalizer=_normalizer(data_source)))

def _default_compute_score_format_answer(data_source, solution_str, ground_truth, extra_info=None):
    from . import qa_em_and_format
    return _to_float(qa_em_and_format.compute_score_format_answer(
        solution_str, ground_truth, normalizer=_normalizer(data_source)))

def _default_compute_score(data_source, solution_str, ground_truth, extra_info=None):
    from . import qa_em_and_format
    out = {
        "score": _default_compute_score_format_answer(data_source, solution_str, ground_truth, extra_info),
        "em": _default_compute_score_answer(data_source, solution_str, ground_truth, extra_info),
        "sm": _default_compute_score_sm(data_source, solution_str, ground_truth, extra_info),
        "f1": _default_compute_score_f1(data_source, solution_str, ground_truth, extra_info),
        "format": _default_compute_score_format(data_source, solution_str, extra_info),
    }

    out.update(qa_em_and_format.compute_score_format_parts(solution_str))
    return out
