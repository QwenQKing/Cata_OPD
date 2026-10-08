import torch
from typing import Dict, Tuple, List
from dataclasses import dataclass

@dataclass
class TensorConfig:
    pad_token_id: int

class TensorHelper:
    def __init__(self, config: TensorConfig):
        self.config = config

    def cut_to_effective_len(self, tensor_dict: Dict[str, torch.Tensor],
                             keys: List[str], cut_left: bool = True) -> Dict[str, torch.Tensor]:

        effective_len = tensor_dict['attention_mask'].sum(dim=1).max()
        result = tensor_dict.copy()

        for key in keys:
            if cut_left:
                result[key] = tensor_dict[key][:, -effective_len:]
            else:
                result[key] = tensor_dict[key][:, :effective_len]
        return result

    def convert_pad_structure(self, tensor: torch.Tensor, pad_to_left: bool = True) -> torch.Tensor:

        mask = tensor != self.config.pad_token_id if pad_to_left else tensor == self.config.pad_token_id

        sorted_indices = mask.to(torch.int64).argsort(dim=1, stable=True)
        sorted_tensor = tensor.gather(1, sorted_indices)

        effective_len = (tensor != self.config.pad_token_id).sum(dim=1).max().item()

        if pad_to_left:
            return sorted_tensor[:, -effective_len:]
        else:
            return sorted_tensor[:, :effective_len]

    def create_attention_mask(self, input_ids: torch.Tensor) -> torch.Tensor:

        return torch.where(input_ids != self.config.pad_token_id, 1, 0)

    def create_position_ids(self, attention_mask: torch.Tensor) -> torch.Tensor:

        return (torch.cumsum(attention_mask, dim=1) - 1) * attention_mask

    def concatenate_with_padding(self, tensors: List[torch.Tensor],
                                pad_to_left: bool = True) -> torch.Tensor:

        concatenated = torch.cat(tensors, dim=1)
        padded_tensor = self.convert_pad_structure(concatenated, pad_to_left)
        return padded_tensor
