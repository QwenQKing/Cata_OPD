pass

import importlib.metadata
import sys
from functools import lru_cache
from typing import Optional

import torch
from packaging import version
from transformers.modeling_flash_attention_utils import _flash_attention_forward
from transformers.modeling_utils import PreTrainedModel

from verl.utils.ulysses import (
    gather_heads_scatter_seq,
    gather_seq_scatter_heads,
    get_ulysses_sequence_parallel_group,
    get_ulysses_sequence_parallel_world_size,
)

def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    pass
    batch, slen, num_key_value_heads, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, :, None, :].expand(batch, slen, num_key_value_heads, n_rep, head_dim)
    return hidden_states.reshape(batch, slen, num_key_value_heads * n_rep, head_dim)

def _ulysses_flash_attention_forward(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    *args,
    position_ids: Optional[torch.Tensor] = None,
    **kwargs,
):
    pass
    ulysses_sp_size = get_ulysses_sequence_parallel_world_size()

    if ulysses_sp_size > 1:
        assert position_ids is not None, "position_ids is required for Ulysses sequence parallelism"

        repeats = max(ulysses_sp_size // key_states.size(2), 1)
        key_states = repeat_kv(key_states, repeats)
        value_states = repeat_kv(value_states, repeats)

        query_states = gather_seq_scatter_heads(query_states, seq_dim=1, head_dim=2)
        key_states = gather_seq_scatter_heads(key_states, seq_dim=1, head_dim=2)
        value_states = gather_seq_scatter_heads(value_states, seq_dim=1, head_dim=2)

        position_ids_list = [torch.empty_like(position_ids) for _ in range(ulysses_sp_size)]
        torch.distributed.all_gather(position_ids_list, position_ids, group=get_ulysses_sequence_parallel_group())
        position_ids = torch.concat(position_ids_list, dim=-1)

    attn_output = _flash_attention_forward(query_states, key_states, value_states, *args, position_ids=position_ids, **kwargs)

    if ulysses_sp_size > 1:
        
        attn_output = gather_heads_scatter_seq(attn_output, seq_dim=1, head_dim=2)

    return attn_output

def apply_monkey_patch(model: PreTrainedModel, ulysses_sp_size: int):
    pass
    module = sys.modules[model.__module__]

    num_attention_heads, num_key_value_heads = model.config.num_attention_heads, model.config.num_key_value_heads
    assert num_attention_heads % ulysses_sp_size == 0, f"num_attention_heads {num_attention_heads} must be divisible by ulysses_sp_size {ulysses_sp_size}"
    assert num_key_value_heads % ulysses_sp_size == 0 or ulysses_sp_size % num_key_value_heads == 0, (
        f"num_key_value_heads {num_key_value_heads} must be divisible by ulysses_sp_size {ulysses_sp_size}or vise versa. Upon ulysses_sp_size % num_key_value_heads == 0,kv heads are repeated to ensure correctness."
    )
    
    if model.config.model_type in ("qwen2_vl", "qwen2_5_vl"):  
        from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLFlashAttention2
        from transformers.models.qwen2_vl.modeling_qwen2_vl import Qwen2VLFlashAttention2

        from verl.models.transformers.qwen2_vl import ulysses_flash_attn_forward

        Qwen2VLFlashAttention2.forward = ulysses_flash_attn_forward
        Qwen2_5_VLFlashAttention2.forward = ulysses_flash_attn_forward
        print("Monkey patch FlashAttention2.forward in Qwen2VL")
        return

    if hasattr(module, "_flash_attention_forward"):
        module._flash_attention_forward = _ulysses_flash_attention_forward
        print(f"Monkey patch _flash_attention_forward in {model.__module__}")
    else:
        
        from transformers.integrations import flash_attention

        flash_attention._flash_attention_forward = _ulysses_flash_attention_forward
        print(f"Monkey patch _flash_attention_forward in {flash_attention.__name__}")

@lru_cache
def is_transformers_version_in_range(min_version: str, max_version: str) -> bool:
    try:
        
        transformers_version = importlib.metadata.version("transformers")
    except importlib.metadata.PackageNotFoundError as e:
        raise ModuleNotFoundError("The `transformers` package is not installed.") from e

    return version.parse(min_version) <= version.parse(transformers_version) <= version.parse(max_version)
