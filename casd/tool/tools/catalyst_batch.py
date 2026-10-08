from typing import Callable, Dict, List, Union
import asyncio

from casd.prompts import build_solve_prompt, wrap_guidance
from casd.tool.tools.teacher_review import TeacherReviewer, verify

Messages = List[Dict[str, str]]

def catalyst_batch(items: List[Dict],
                   batch_student_fn: Callable[[List[Messages]], List[str]],
                   max_rounds: int = 3) -> List[Dict]:

    if not items:
        return []

    states = []
    for it in items:
        q, gt, y0 = it["question"], it["ground_truth"], it["y0"]
        ds = it.get("data_source", "hotpotqa/hotpot_qa")
        msgs: Messages = [{"role": "user", "content": build_solve_prompt(q, deploy=True)},
                          {"role": "assistant", "content": y0}]
        states.append({"q": q, "gt": gt, "ds": ds, "messages": msgs,
                       "teacher": TeacherReviewer(q, gt), "last": y0,
                       "done": False, "y1": None, "rounds": 0, "guidances": []})

    for r in range(1, max_rounds + 1):
        active = [s for s in states if not s["done"]]
        if not active:
            break

        async def _guide_all(batch):
            return await asyncio.gather(*[s["teacher"].guide_async(s["last"]) for s in batch])
        guidances = asyncio.run(_guide_all(active))

        ans_states: List[Dict] = []
        to_answer: List[Messages] = []
        for s, g in zip(active, guidances):
            if not g:
                s["done"] = True
                continue
            s["guidances"].append(g)
            s["messages"].append({"role": "user", "content": wrap_guidance(g)})

            _compact = [s["messages"][0],
                        {"role": "assistant", "content": s["last"]},
                        {"role": "user", "content": wrap_guidance(g)}]
            to_answer.append(_compact)
            ans_states.append(s)
        if not to_answer:
            continue
        print(f"[DBG-CASD] catalyst round={r}: #to_answer={len(to_answer)} "
              f"喂模型压缩上下文[0]_roles={[m['role'] for m in to_answer[0]]}(应=3) "
              f"存档全量长度={len(ans_states[0]['messages'])}(随轮增长)", flush=True)

        answers = batch_student_fn(to_answer)

        for s, a in zip(ans_states, answers):
            s["messages"].append({"role": "assistant", "content": a})
            s["last"] = a
            s["rounds"] = r
            if verify(a, s["gt"], s["ds"]):
                s["y1"] = a
                s["done"] = True

    return [{"y1": s["y1"], "has_y1": s["y1"] is not None, "rounds": s["rounds"],
             "guidances": s["guidances"], "history": s["messages"],
             "teacher_calls": s["teacher"].n_calls,
             "teacher_ok": s["teacher"].n_ok, "teacher_fail": s["teacher"].n_fail,
             "teacher_in_tokens": s["teacher"].in_tokens, "teacher_out_tokens": s["teacher"].out_tokens}
            for s in states]
