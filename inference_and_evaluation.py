import os
import json
import re
import time
import collections
from pathlib import Path
from datetime import datetime
from typing import List
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
from openai import OpenAI

from casd.src.reward_score import _default_compute_score
import casd.vllm_infer.config as cfg

if not cfg.OPENAI_API_BASE:
    raise RuntimeError("VLLM_API_BASE is required")

MAX_TOKENS = int(os.environ.get("EVAL_MAX_TOKENS", "8192"))
BATCH_SIZE = int(os.environ.get("EVAL_BATCH", "64"))
CONCURRENCY = int(os.environ.get("EVAL_CONCURRENCY", "64"))
EVAL_TEMP = float(os.environ.get("EVAL_TEMP", cfg.TEMPERATURE))
INPUT_DIR = Path(os.environ.get("EVAL_INPUT_DIR", "dataset/eval"))

_TAG = os.environ.get("EVAL_TAG", "eval")
_TS = datetime.now().strftime("%Y%m%d-%H%M%S")
OUTPUT_DIR = Path(os.environ.get("EVAL_OUTPUT_DIR")
                  or f"results/{_TAG}_temp{EVAL_TEMP:g}_{_TS}")

PASS_K = sorted(int(x) for x in os.environ.get("PASS_K", "").replace(" ", "").split(",") if x)
K_MAX = max(PASS_K) if PASS_K else 1
EVAL_LIMIT = int(os.environ.get("EVAL_LIMIT", "0"))

QUAL_KEYS = ([f"pass@{k}" for k in PASS_K] + [f"maj@{k}" for k in PASS_K]) if PASS_K \
            else ["em", "f1", "format", "score"]

def _json_safe(obj):

    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(x) for x in obj]
    return obj

def _gen(client, p: str, params, n: int) -> dict:

    t0 = time.time()
    try:
        r = client.chat.completions.create(
            model=params["model"],
            messages=[{"role": "user", "content": p}],
            temperature=params["temperature"],
            top_p=params["top_p"],
            max_tokens=params["max_tokens"],
            n=n,
            extra_body={"repetition_penalty": params["rep_penalty"]},
        )
        dt = time.time() - t0
        samples = [(c.message.content or "") for c in (r.choices or [])]
        u = getattr(r, "usage", None)
        return {"samples": samples,
                "in_tokens": int(u.prompt_tokens) if u else 0,
                "out_tokens": int(u.completion_tokens) if u else 0,
                "total_tokens": int(u.total_tokens) if u else 0,
                "latency_s": dt}
    except Exception as e:
        print(f"[ERROR] generate failed: {e}")
        return {"samples": [""] * n, "in_tokens": 0, "out_tokens": 0, "total_tokens": 0, "latency_s": time.time() - t0}

def generate(client, prompts: List[str], params, n: int) -> List[dict]:

    total = len(prompts)
    results = [None] * total
    done = 0
    with ThreadPoolExecutor(max_workers=CONCURRENCY) as ex:
        futs = {ex.submit(_gen, client, p, params, n): i for i, p in enumerate(prompts)}
        for fut in as_completed(futs):
            results[futs[fut]] = fut.result()
            done += 1
            if done % 20 == 0 or done == total:
                print(f"      并发完成 {done}/{total}", end="\r", flush=True)
    print()
    return results

def score_one(data_source: str, response: str, ground_truth) -> dict:

    solution_str = f"<|im_start|>assistant\n{response}<|im_end|>"
    try:
        return _default_compute_score(data_source, solution_str, ground_truth)
    except Exception as e:
        print(f"[WARN] score failed (data_source={data_source}): {e}")
        return {"score": 0.0, "em": 0.0, "sm": 0.0, "f1": 0.0, "format": 0.0}

def process_parquet(path: Path, client, params):
    df = pd.read_parquet(path)
    if EVAL_LIMIT > 0:
        df = df.iloc[:EVAL_LIMIT]
    name = path.stem
    fdir = OUTPUT_DIR / name
    fdir.mkdir(parents=True, exist_ok=True)
    results = []

    t0 = time.time()
    log_path = fdir / "log.txt"
    with open(log_path, "w", encoding="utf-8") as log:
        log.write(f"文件: {path.name}\n时间: {datetime.now():%Y-%m-%d %H:%M:%S}\n总数: {len(df)}\n"
                  + (f"PASS_K={PASS_K}(每题采 {K_MAX} 样本, temp {EVAL_TEMP:g})\n" if PASS_K else "")
                  + "=" * 80 + "\n")
        for start in range(0, len(df), BATCH_SIZE):
            bdf = df.iloc[start:start + BATCH_SIZE]
            prompts, gts, dss, idxs = [], [], [], []
            for idx, row in bdf.iterrows():
                prompts.append(row["prompt"][0]["content"])
                gt = _json_safe(row["reward_model"]["ground_truth"])
                gts.append(gt if isinstance(gt, list) else [gt])
                dss.append(str(row["data_source"]))
                idxs.append(int(idx))

            outs = generate(client, prompts, params, K_MAX)

            for i, o in enumerate(outs):
                samples = o["samples"]
                rec = {"data_id": idxs[i], "data_source": dss[i], "prompt": prompts[i], "ground_truth": gts[i],
                       "in_tokens": o["in_tokens"], "out_tokens": o["out_tokens"],
                       "total_tokens": o["total_tokens"], "latency_s": round(o["latency_s"], 3)}
                if PASS_K:
                    answers, ems = [], []
                    for s in samples:
                        a = re.findall(r"<answer>(.*?)</answer>", s, re.DOTALL)
                        answers.append(a[0].strip() if a else "")
                        ems.append(int(score_one(dss[i], s, gts[i])["em"]))
                    rec["samples"], rec["sample_answers"], rec["sample_em"] = samples, answers, ems
                    for k in PASS_K:
                        rec[f"pass@{k}"] = 1.0 if any(ems[:k]) else 0.0
                        cnt = collections.Counter(answers[:k])
                        mode = cnt.most_common(1)[0][0] if cnt else ""
                        rec[f"maj@{k}"] = float(next((e for a, e in zip(answers[:k], ems[:k]) if a == mode), 0))
                    log.write(f"\n{'#'*80}\nID {idxs[i]} | {dss[i]}\n[输入 prompt]\n{prompts[i]}\n\n"
                              f"[{len(samples)} 样本答案] {answers}\n[各样本对错] {ems}\n[标准答案] {gts[i]}\n"
                              + "  ".join(f"pass@{k}={rec[f'pass@{k}']:.0f}" for k in PASS_K) + "  |  "
                              + "  ".join(f"maj@{k}={rec[f'maj@{k}']:.0f}" for k in PASS_K)
                              + f"\nin={o['in_tokens']} out={o['out_tokens']}(共{len(samples)}样本) latency={o['latency_s']:.2f}s\n")
                else:
                    s0 = samples[0] if samples else ""
                    a = re.findall(r"<answer>(.*?)</answer>", s0, re.DOTALL)
                    sc = score_one(dss[i], s0, gts[i])
                    rec["response"] = s0
                    rec["predicted_answer"] = a[0].strip() if a else ""
                    rec.update({k: round(float(v), 2) for k, v in sc.items()})
                    log.write(f"\n{'#'*80}\nID {idxs[i]} | {dss[i]}\n[输入 prompt]\n{prompts[i]}\n\n"
                              f"[模型完整输出]\n{s0}\n\n[抽取答案] {rec['predicted_answer']}\n[标准答案] {gts[i]}\n"
                              f"em={sc['em']:.2f} f1={sc['f1']:.2f} format={sc['format']:.2f} score={sc['score']:.2f} "
                              f"| in={o['in_tokens']} out={o['out_tokens']} total={o['total_tokens']} latency={o['latency_s']:.2f}s\n")
                results.append(rec)
            log.flush()
            print(f"  {name}: {min(start + BATCH_SIZE, len(df))}/{len(df)} done")

    wall = time.time() - t0
    (fdir / "res.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    return _summarize(name, results, wall)

def _summarize(name: str, results: List[dict], wall: float = 0.0) -> dict:

    by_src = collections.defaultdict(list)
    for r in results:
        by_src[r["data_source"]].append(r)
    print(f"\n===== {name} 汇总 =====")
    out = {}
    for src, rows in sorted(by_src.items()):
        n = len(rows)
        mean = lambda k: sum(r[k] for r in rows) / n
        out_tok = sum(r["out_tokens"] for r in rows)
        d = {"n": n}
        for qk in QUAL_KEYS:
            d[qk] = round(mean(qk), 2)
        d.update({"avg_in_tokens": round(mean("in_tokens"), 1),
                  "avg_out_tokens": round(mean("out_tokens"), 1),
                  "avg_total_tokens": round(mean("total_tokens"), 1),
                  "avg_latency_s": round(mean("latency_s"), 2),
                  "throughput_tok_s": round(out_tok / wall, 1) if wall > 0 else 0.0,
                  "wall_s": round(wall, 1)})

        _acc = d.get("em", d.get("pass@1", 0.0)); _ao = d["avg_out_tokens"]
        d["acc_per_1k_tok"] = round(_acc / (_ao / 1000.0), 3) if _ao > 0 else 0.0
        d["tok_per_correct"] = round(_ao / _acc, 1) if _acc > 0 else None
        out[src] = d
        qual = "  ".join(f"{qk}={d[qk]:.2f}" for qk in QUAL_KEYS)
        print(f"  [{src}] n={n}  {qual}  | tok {d['avg_in_tokens']:.0f}/{d['avg_out_tokens']:.0f}"
              f"  lat {d['avg_latency_s']:.2f}s  {d['throughput_tok_s']:.0f} tok/s")
    return out

def main():
    mode = f"pass@k {PASS_K}(每题 {K_MAX} 样本)" if PASS_K else "单样本"
    print(f"[评测] 模式={mode} | 输入 {INPUT_DIR.name} | 温度 {EVAL_TEMP:g} | 并发 {CONCURRENCY}"
          + (f" | 每集限 {EVAL_LIMIT} 条" if EVAL_LIMIT else "") + f" | 输出 {OUTPUT_DIR.name}")
    client = OpenAI(api_key=cfg.OPENAI_API_KEY, base_url=cfg.OPENAI_API_BASE)
    params = {
        "model": cfg.MODEL_NAME, "temperature": EVAL_TEMP,
        "top_p": cfg.TOP_P, "max_tokens": MAX_TOKENS, "rep_penalty": cfg.REPETITION_PENALTY,
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    files = sorted(INPUT_DIR.glob("*.parquet"))
    if not files:
        print(f"{INPUT_DIR} 下没有 parquet（待评测集需含 prompt/reward_model/data_source 列）")
        return
    print(f"找到 {len(files)} 个 parquet")
    all_summ = {}
    for f in files:
        print(f"\n处理 {f}")
        try:
            s = process_parquet(f, client, params)
            if s:
                all_summ.update(s)
        except Exception as e:
            print(f"[ERROR] 处理 {f} 失败: {e}")

    if all_summ:
        METRICS = QUAL_KEYS + ["avg_in_tokens", "avg_out_tokens", "avg_total_tokens",
                               "avg_latency_s", "throughput_tok_s"]
        avg = {m: round(sum(v[m] for v in all_summ.values()) / len(all_summ), 2) for m in METRICS}
        avg["total_samples"] = sum(v["n"] for v in all_summ.values())

        _acc = avg.get("em", avg.get("pass@1", 0.0))
        avg["acc_per_1k_tok"] = round(_acc / (avg["avg_out_tokens"] / 1000.0), 3) if avg["avg_out_tokens"] > 0 else 0.0
        avg["tok_per_correct"] = round(avg["avg_out_tokens"] / _acc, 1) if _acc > 0 else None
        summary = {"output_dir": OUTPUT_DIR.name, "input_dir": INPUT_DIR.name,
                   "temperature": EVAL_TEMP, "tag": _TAG, "timestamp": _TS, "pass_k": PASS_K,
                   "served_model": cfg.MODEL_NAME, "n_datasets": len(all_summ),
                   "average": avg, "per_dataset": all_summ}
        (OUTPUT_DIR / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n===== 全部 {len(all_summ)} 集平均 =====")
        print("  " + "  ".join(f"{qk}={avg[qk]:.2f}" for qk in QUAL_KEYS))
        print(f"  平均 token 入/出/总={avg['avg_in_tokens']:.0f}/{avg['avg_out_tokens']:.0f}/{avg['avg_total_tokens']:.0f}"
              f"  | 平均延迟={avg['avg_latency_s']:.2f}s  | 吞吐≈{avg['throughput_tok_s']:.0f} tok/s")
        print(f"  效率: 准确率/1k_tok={avg['acc_per_1k_tok']:.3f}  | 每答对≈{avg['tok_per_correct']} out_tok")
        print(f"规范汇总已写: {OUTPUT_DIR.name}/summary.json")
    print("\n全部完成")

if __name__ == "__main__":
    main()
