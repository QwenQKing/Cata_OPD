def _default_compute_score(data_source, solution_str, ground_truth, extra_info=None):
    if data_source.rsplit("/", 1)[-1] == "gsm8k":
        from . import gsm8k

        res = gsm8k.compute_score(solution_str, ground_truth)
    elif data_source.rsplit("/", 1)[-1] in ["MATH", "MATH-lighteval"]:
        from . import math

        res = math.compute_score(solution_str, ground_truth)

    elif data_source == "math_dapo" or data_source.startswith("aime"):
        from . import math_dapo

        res = math_dapo.compute_score(solution_str, ground_truth)
    elif data_source in [
        "numina_aops_forum",
        "numina_synthetic_math",
        "numina_amc_aime",
        "numina_synthetic_amc",
        "numina_cn_k12",
        "numina_olympiads",
    ]:
        from . import prime_math

        res = prime_math.compute_score(solution_str, ground_truth)
    elif data_source in ["codecontests", "apps", "codeforces", "taco"]:
        from . import prime_code

        res = prime_code.compute_score(solution_str, ground_truth, continuous=True)
    elif data_source.rsplit("/", 1)[-1] == "geometry3k":
        from . import geo3k

        res = geo3k.compute_score(solution_str, ground_truth)
    else:
        raise NotImplementedError(f"Reward function is not implemented for {data_source=}")

    if isinstance(res, dict):
        return res
    elif isinstance(res, (int, float, bool)):
        return float(res)
    else:
        return float(res[0])
