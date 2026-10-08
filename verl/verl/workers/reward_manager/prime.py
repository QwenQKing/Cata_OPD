import asyncio
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from typing import Callable, Optional

import torch
from transformers import PreTrainedTokenizer

from verl import DataProto
from verl.utils.reward_score import _default_compute_score

async def single_compute_score(evaluation_func, completion, reference, task, task_extra_info, executor, timeout=300.0):
    loop = asyncio.get_running_loop()
    try:
        
        tasks = [
            asyncio.wait_for(
                loop.run_in_executor(
                    executor,
                    partial(evaluation_func, task, completion, reference, task_extra_info),  
                ),
                timeout=timeout,
            )
        ]
        return await asyncio.gather(*tasks)
    except asyncio.TimeoutError:
        print(f"Timeout occurred for completion: {completion}")
        return None  
    except Exception as e:
        print(f"Error processing completion: {completion[:10]}, Error: {e}")
        return None  

async def parallel_compute_score_async(evaluation_func, completions, references, tasks, extra_info=None, num_processes=64):
    scores = []
    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        if extra_info is None:
            extra_info = [None] * len(tasks)
        
        tasks_async = [single_compute_score(evaluation_func, completion, reference, task, task_extra_info, executor, timeout=300.0) for completion, reference, task, task_extra_info in zip(completions, references, tasks, extra_info)]
        
        try:
            results = await asyncio.gather(*tasks_async, return_exceptions=False)
        except:
            for pid, proc in executor._processes.items():
                try:
                    proc.kill()
                except Exception as kill_err:
                    print("shut down failed: " + str(kill_err))
            raise

    for result, completion, reference, task in zip(results, completions, references, tasks):
        if isinstance(result, Exception) or result is None:
            
            scores.append(0.0)
        elif isinstance(result[0], (int, float, bool)):
            scores.append(float(result[0]))
        else:
            scores.append(float(result[0][0]))
    return scores

class PrimeRewardManager:
    pass

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        num_examine: int,
        compute_score: Optional[Callable] = None,
        reward_fn_key: str = "data_source",
    ) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine  
        self.compute_score = compute_score or _default_compute_score
        self.reward_fn_key = reward_fn_key

    def verify(self, data):
        pass
        
        prompt_ids = data.batch["prompts"]

        response_ids = data.batch["responses"]
        sequences_str = self.tokenizer.batch_decode(response_ids, skip_special_tokens=True)
        ground_truth = [data_item.non_tensor_batch["reward_model"]["ground_truth"] for data_item in data]
        data_sources = data.non_tensor_batch[self.reward_fn_key]
        extra_info = data.non_tensor_batch.get("extra_info", None)

        assert len(sequences_str) == len(ground_truth) == len(data_sources)
        try:
            scores = asyncio.run(
                parallel_compute_score_async(
                    self.compute_score,
                    sequences_str,
                    ground_truth,
                    data_sources,
                    extra_info=extra_info,
                    num_processes=64,
                )
            )
        except asyncio.TimeoutError:
            print("Global timeout in reward computing! Setting all as 0.")
            scores = [0.0 for _ in range(len(sequences_str))]
        except Exception as e:
            print(f"Unexpected error in batched reward computing. Setting all as 0.: {e}")
            scores = [0.0 for _ in range(len(sequences_str))]
        data.batch["acc"] = torch.tensor(scores, dtype=torch.float32, device=prompt_ids.device)
        return scores

    def __call__(self, data: DataProto, return_dict: bool = False):
        pass

        if "rm_scores" in data.batch.keys():
            return data.batch["rm_scores"]

        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)

        already_print_data_sources = {}

        prompt_ids = data.batch["prompts"]
        prompt_length = prompt_ids.shape[-1]

        response_ids = data.batch["responses"]
        valid_response_length = data.batch["attention_mask"][:, prompt_length:].sum(dim=-1)
        sequences_str = self.tokenizer.batch_decode(response_ids, skip_special_tokens=True)
        data_sources = data.non_tensor_batch["data_source"]

        scores = self.verify(data)

        for i in range(len(data)):
            data_source = data_sources[i]
            reward_tensor[i, valid_response_length[i].item() - 1] = scores[i]

            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                print(sequences_str)

        if return_dict:
            return {"reward_tensor": reward_tensor}
        else:
            return reward_tensor
