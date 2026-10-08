import os
import re
from collections import defaultdict

import torch

from verl import DataProto
from casd.src.reward_score import _default_compute_score

class AgentRewardManager:

    def __init__(self, tokenizer, num_examine, compute_score=None, reward_fn_key="data_source") -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.compute_score = compute_score or _default_compute_score
        self.reward_fn_key = reward_fn_key

        self._call_n = 0
        self.train_log_file = os.environ.get("TRAIN_LOG_FILE", "train_log.txt")
        self.train_log_n = int(os.environ.get("TRAIN_LOG_N", "10"))

    def _log_case(self, f, step, idx, sequences_str, ground_truth, data_source, score):

        from casd.src.reward_score.qa_em_and_format import extract_solution, has_answer_tags
        blocks = re.findall(r'<\|im_start\|>assistant\n(.*?)<\|im_end\|>', sequences_str, re.DOTALL)
        block = blocks[-1] if blocks else ""
        answer = extract_solution(block)
        gt = list(ground_truth) if not isinstance(ground_truth, str) else [ground_truth]
        d = score if isinstance(score, dict) else {"score": score}
        f.write("=" * 110 + "\n")
        f.write(f"STEP {step} | 第 {idx} 条 | data_source = {data_source}\n")
        f.write("-" * 110 + "\n【完整输入 solution_str(prompt+response，含特殊标记)】\n")
        f.write(sequences_str.rstrip() + "\n")
        f.write("-" * 110 + "\n【reward 计算全过程】\n")
        f.write(f"  打分所用 assistant 块(y0) : 有 <answer> 标签? {has_answer_tags(block)}\n")
        f.write(f"  抽取的学生答案 <answer>    : {answer!r}\n")
        f.write(f"  ground_truth(标准答案)    : {gt}\n")
        f.write(f"  格式分 format             : {d.get('format')}   (4×0.25: <think>标签 + think实质≥30字符 + <answer>标签 + answer非空)\n")
        f.write(f"  答案判定 em = math_equal(<answer>, ground_truth) : {d.get('em')}   (1=数学等价/0=不等)\n")
        f.write(f"  最终 reward score         : {d.get('score')}   (格式分==1 时 score = -1 + format + em ∈ {{0,1}})\n")
        f.write(f"  其它诊断指标              : sm={d.get('sm')}  f1={d.get('f1')}\n")
        f.write("=" * 110 + "\n\n")

    def __call__(self, data: DataProto, return_dict=False):

        if "rm_scores" in data.batch.keys():
            if return_dict:
                return {"reward_tensor": data.batch["rm_scores"]}
            else:
                return data.batch["rm_scores"]

        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)

        reward_extra_info = defaultdict(list)

        already_print_data_sources = {}

        self._call_n += 1
        log_f, logged = None, 0
        if self.train_log_file and self.train_log_file.lower() not in ("off", "none", ""):
            try:
                log_f = open(self.train_log_file, "a", encoding="utf-8")
                log_f.write(f"\n########## STEP {self._call_n}  (batch={len(data)} 条,记前 {self.train_log_n} 条) ##########\n")
            except Exception as e:
                print(f"[reward-log] 打开 {self.train_log_file} 失败: {e}")
                log_f = None

        for i in range(len(data)):
            data_item = data[i]

            prompt_ids = data_item.batch["prompts"]

            prompt_length = prompt_ids.shape[-1]

            valid_prompt_length = data_item.batch["attention_mask"][:prompt_length].sum()

            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch["responses"]

            valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()

            valid_response_ids = response_ids[:valid_response_length]

            sequences = torch.cat((valid_prompt_ids, valid_response_ids))

            sequences_str = self.tokenizer.decode(sequences, skip_special_tokens=False)
            pad_token_id = self.tokenizer.pad_token_id

            sequences_str = sequences_str.split(self.tokenizer.decode([pad_token_id]))[0]

            if not sequences_str.endswith(self.tokenizer.eos_token):
                sequences_str += self.tokenizer.eos_token

            ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]

            data_source = data_item.non_tensor_batch[self.reward_fn_key]

            extra_info = data_item.non_tensor_batch.get("extra_info", None)

            score = self.compute_score(
                data_source=data_source,
                solution_str=sequences_str,
                ground_truth=ground_truth,
                extra_info=extra_info,
            )

            if isinstance(score, dict):
                reward = score["score"]

                score["think_tokens"] = 0.0
                try:
                    from casd.src.reward_score.qa_em_and_format import extract_think_text
                    _tt = extract_think_text(sequences_str)
                    if _tt:
                        score["think_tokens"] = float(len(self.tokenizer(_tt, add_special_tokens=False)["input_ids"]))
                except Exception:
                    pass

                for key, value in score.items():
                    reward_extra_info[key].append(value)
            else:

                reward = score

            reward_tensor[i, valid_response_length - 1] = reward

            if log_f is not None and logged < self.train_log_n:
                logged += 1
                try:
                    self._log_case(log_f, self._call_n, logged, sequences_str, ground_truth, data_source, score)
                except Exception as e:
                    print(f"[reward-log] 写第 {logged} 条失败: {e}")

            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1

                print("[prompt+response]", sequences_str)

                print("[ground_truth]", ground_truth)

                if isinstance(score, dict):
                    for key, value in score.items():
                        print(f"[{key}]", value)
                else:
                    print("[score]", score)

        if log_f is not None:
            log_f.flush()
            log_f.close()

        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": reward_extra_info
            }
        else:
            return reward_tensor
