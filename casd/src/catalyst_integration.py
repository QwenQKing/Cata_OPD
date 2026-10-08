from typing import Dict, List, Tuple

import numpy as np
import torch

def select_wrong_items(batch, reward_extra_infos_dict, tokenizer) -> Tuple[List[Dict], List[int]]:

    em = reward_extra_infos_dict.get("em", None)
    if em is None:
        return [], []
    responses = batch.batch["responses"]
    items: List[Dict] = []
    indices: List[int] = []
    for i in range(len(em)):
        if float(em[i]) >= 1.0:
            continue
        extra = batch.non_tensor_batch["extra_info"][i]
        question = extra["question"] if isinstance(extra, dict) else str(extra)
        ground_truth = batch.non_tensor_batch["reward_model"][i]["ground_truth"]

        data_source = (batch.non_tensor_batch["data_source"][i]
                       if "data_source" in batch.non_tensor_batch else "gsm8k")
        y0 = tokenizer.decode(responses[i], skip_special_tokens=True).strip()
        items.append({"question": question, "ground_truth": ground_truth, "y0": y0,
                      "data_source": str(data_source)})
        indices.append(i)
    return items, indices

def select_allwrong_groups(batch, reward_extra_infos_dict, tokenizer) -> Tuple[List[Dict], List[int], Dict]:

    from collections import defaultdict
    em = reward_extra_infos_dict.get("em", None)
    if em is None:
        return [], [], {"groups": 0, "allwrong": 0, "mixed": 0}
    responses = batch.batch["responses"]
    groups = defaultdict(list)
    for i in range(len(em)):
        extra = batch.non_tensor_batch["extra_info"][i]
        q = extra["question"] if isinstance(extra, dict) else str(extra)
        groups[q].append(i)
    items: List[Dict] = []
    indices: List[int] = []
    n_allwrong = 0
    for q, idxs in groups.items():
        if all(float(em[i]) < 1.0 for i in idxs):
            n_allwrong += 1
            i0 = idxs[0]
            ground_truth = batch.non_tensor_batch["reward_model"][i0]["ground_truth"]
            data_source = (batch.non_tensor_batch["data_source"][i0]
                           if "data_source" in batch.non_tensor_batch else "gsm8k")
            y0 = tokenizer.decode(responses[i0], skip_special_tokens=True).strip()
            items.append({"question": q, "ground_truth": ground_truth, "y0": y0,
                          "data_source": str(data_source)})
            indices.append(i0)
    stats = {"groups": len(groups), "allwrong": n_allwrong, "mixed": len(groups) - n_allwrong}
    return items, indices, stats

def best_of_m_rescue(items, batch_student_fn, m=4):

    from casd.tool.tools.teacher_review import verify
    from casd.prompts.student import build_solve_prompt
    if not items or m <= 0:
        return [{"self_ok": False, "self_y1": None,
                 "history": [{"role": "user", "content": build_solve_prompt(it["question"], deploy=True)}],
                 "m": m, "n_correct": 0} for it in items]

    msgs, owner = [], []
    for j, it in enumerate(items):
        pj = build_solve_prompt(it["question"], deploy=True)
        for _ in range(m):
            msgs.append([{"role": "user", "content": pj}])
            owner.append(j)
    answers = batch_student_fn(msgs)
    best = [None] * len(items)
    ncorr = [0] * len(items)
    attempts = [[] for _ in items]
    for a, j in zip(answers, owner):
        attempts[j].append(a)
        if verify(a, items[j]["ground_truth"], items[j].get("data_source", "gsm8k")):
            ncorr[j] += 1
            if best[j] is None:
                best[j] = a
    results = []
    for j, it in enumerate(items):
        a = best[j]
        results.append({
            "self_ok": a is not None,
            "self_y1": a,
            "history": [{"role": "user", "content": build_solve_prompt(it["question"], deploy=True)},
                        {"role": "assistant", "content": a if a is not None else ""}],
            "m": m, "n_correct": ncorr[j], "attempts": attempts[j],
        })
    _ok = sum(1 for r in results if r["self_ok"])
    print(f"[DBG-CASD] best-of-m 自救(不看GT,m={m}): self_ok={_ok}/{len(items)} → deep_cliff={len(items)-_ok}(交外部老师)", flush=True)
    return results

def _encode_one(tokenizer, prompt_text: str, response_text: str,
                max_prompt_length: int, max_response_length: int):

    p_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    r_ids = tokenizer(response_text, add_special_tokens=False)["input_ids"]

    eos = tokenizer.eos_token_id
    if eos is not None and (len(r_ids) == 0 or r_ids[-1] != eos):
        r_ids = r_ids + [eos]
    p_ids = p_ids[-max_prompt_length:]
    r_ids = r_ids[:max_response_length]
    return p_ids, r_ids

def build_y1_fields(results: List[Dict], indices: List[int], batch_size: int, tokenizer,
                    max_prompt_length: int, max_response_length: int,
                    enable_thinking: bool = True) -> Dict[str, torch.Tensor]:

    pad_id = tokenizer.pad_token_id
    P, R = max_prompt_length, max_response_length
    y1_input_ids = torch.full((batch_size, P + R), pad_id, dtype=torch.long)
    y1_attention_mask = torch.zeros((batch_size, P + R), dtype=torch.long)
    y1_responses = torch.full((batch_size, R), pad_id, dtype=torch.long)
    y1_action_mask = torch.zeros((batch_size, R), dtype=torch.long)
    has_y1 = torch.zeros(batch_size, dtype=torch.float32)

    for res, idx in zip(results, indices):
        if not res.get("has_y1"):
            continue

        user_content = res["history"][0]["content"]
        prompt_text = tokenizer.apply_chat_template(
            [{"role": "user", "content": user_content}],
            add_generation_prompt=True,
            tokenize=False,
            enable_thinking=enable_thinking,
        )
        p_ids, r_ids = _encode_one(tokenizer, prompt_text, res["y1"], P, R)
        lp, lr = len(p_ids), len(r_ids)
        if lr == 0:
            continue

        y1_input_ids[idx, P - lp:P] = torch.tensor(p_ids, dtype=torch.long)
        y1_attention_mask[idx, P - lp:P] = 1

        y1_input_ids[idx, P:P + lr] = torch.tensor(r_ids, dtype=torch.long)
        y1_attention_mask[idx, P:P + lr] = 1

        y1_responses[idx, :lr] = torch.tensor(r_ids, dtype=torch.long)
        y1_action_mask[idx, :lr] = 1
        has_y1[idx] = 1.0

    y1_position_ids = (torch.cumsum(y1_attention_mask, dim=-1) - 1).clamp(min=0)

    return {
        "y1_input_ids": y1_input_ids,
        "y1_attention_mask": y1_attention_mask,
        "y1_position_ids": y1_position_ids,
        "y1_responses": y1_responses,
        "y1_action_mask": y1_action_mask,
        "has_y1": has_y1,
    }

def build_y1_aug_fields(results: List[Dict], indices: List[int], batch_size: int, tokenizer,
                        max_prompt_length: int, max_response_length: int,
                        enable_thinking: bool = True) -> Dict[str, torch.Tensor]:

    pad_id = tokenizer.pad_token_id
    P, R = max_prompt_length, max_response_length
    y1aug_input_ids = torch.full((batch_size, P + R), pad_id, dtype=torch.long)
    y1aug_attention_mask = torch.zeros((batch_size, P + R), dtype=torch.long)

    for res, idx in zip(results, indices):
        if not res.get("has_y1"):
            continue
        history = res.get("history", [])
        if len(history) < 2:
            continue

        q_user = history[0]["content"]
        guides = [m["content"] for m in history[1:-1] if m.get("role") == "user"]
        hint = guides[-1] if guides else ""
        aug_user = q_user + ("\n\n" + hint if hint else "")

        if hint and not globals().get("_AUG_DBG_DONE"):
            globals()["_AUG_DBG_DONE"] = True
            _cjk = [c for c in aug_user if "一" <= c <= "鿿"]
            print(f"[AUG-DBG] barrier 上下文 x̃=[q;引导] 自检 | 含中文={bool(_cjk)}(应False) "
                  f"{('中文字符=' + ''.join(_cjk[:20])) if _cjk else ''} | "
                  f"以<guidance>结尾={aug_user.rstrip().endswith('</guidance>')}(应True) | 长度={len(aug_user)}\n"
                  f"  --- 尾部 300 字符 ---\n  {aug_user[-300:]}", flush=True)
        aug_prompt_text = tokenizer.apply_chat_template(
            [{"role": "user", "content": aug_user}],
            add_generation_prompt=True,
            tokenize=False,
            enable_thinking=enable_thinking,
        )

        ap_ids, ar_ids = _encode_one(tokenizer, aug_prompt_text, res["y1"], P, R)
        lp, lr = len(ap_ids), len(ar_ids)
        if lr == 0:
            continue
        y1aug_input_ids[idx, P - lp:P] = torch.tensor(ap_ids, dtype=torch.long)
        y1aug_attention_mask[idx, P - lp:P] = 1
        y1aug_input_ids[idx, P:P + lr] = torch.tensor(ar_ids, dtype=torch.long)
        y1aug_attention_mask[idx, P:P + lr] = 1

    y1aug_position_ids = (torch.cumsum(y1aug_attention_mask, dim=-1) - 1).clamp(min=0)
    return {
        "y1aug_input_ids": y1aug_input_ids,
        "y1aug_attention_mask": y1aug_attention_mask,
        "y1aug_position_ids": y1aug_position_ids,
    }

def self_rescue_batch(items, batch_student_fn, tokenizer, max_rounds=6):

    from casd.tool.tools.teacher_review import TEACHER_SYSTEM, verify, _leaks, _gt_to_list
    from casd.prompts.student import build_solve_prompt, wrap_guidance
    if not items:
        return []

    states = []
    for it in items:
        states.append({
            "it": it, "last": str(it["y0"]), "done": False, "self_y1": None,
            "guidances": [], "rounds": 0,
            "history": [
                {"role": "user", "content": build_solve_prompt(it["question"], deploy=True)},
                {"role": "assistant", "content": str(it["y0"])},
            ],
        })
    print(f"[DBG-CASD] self_rescue: 让 {len(items)} 个答错样本【自身当 GT-teacher】多轮自救(≤{max_rounds} 轮)", flush=True)
    for r in range(1, int(max_rounds) + 1):
        active = [s for s in states if not s["done"]]
        if not active:
            break

        t_msgs = []
        for s in active:
            gl = _gt_to_list(s["it"]["ground_truth"])
            gt_str = " / ".join(gl) if gl else "(not provided)"

            t_msgs.append([
                {"role": "user", "content": TEACHER_SYSTEM.format(question=s["it"]["question"], ground_truth=gt_str)
                 + "\n\nStudent's most recent response (their reasoning and answer):\n" + s["last"]},
            ])
        guidances = batch_student_fn(t_msgs)

        g_uses, s_msgs = [], []
        for s, g in zip(active, guidances):
            gl = _gt_to_list(s["it"]["ground_truth"])
            g_use = g if (g and not _leaks(g, gl)) else ""
            g_uses.append(g_use)
            s_msgs.append([
                {"role": "user", "content": build_solve_prompt(s["it"]["question"], deploy=True)},
                {"role": "assistant", "content": s["last"]},
                {"role": "user", "content": wrap_guidance(g_use)},
            ])
        answers = batch_student_fn(s_msgs)

        for s, g_use, a in zip(active, g_uses, answers):
            s["rounds"] = r
            s["guidances"].append(g_use)
            s["history"].append({"role": "user", "content": wrap_guidance(g_use)})
            s["history"].append({"role": "assistant", "content": a})
            if verify(a, s["it"]["ground_truth"], s["it"].get("data_source", "gsm8k")):
                s["self_y1"] = a
                s["done"] = True
            else:
                s["last"] = a

    results = []
    for s in states:
        results.append({
            "self_ok": s["self_y1"] is not None,
            "self_y1": s["self_y1"],
            "self_guidance": (s["guidances"][-1] if s["guidances"] else None),
            "history": s["history"],
            "self_rounds": s["rounds"],
        })
    _n_ok = sum(1 for r in results if r["self_ok"])
    print(f"[DBG-CASD] self_rescue: self_ok={_n_ok}/{len(items)} → deep_cliff={len(items)-_n_ok}(交外部老师) | 自救≤{max_rounds}轮", flush=True)
    return results
