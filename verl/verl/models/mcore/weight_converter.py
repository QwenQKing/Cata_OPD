import torch
from megatron.core.transformer import TransformerConfig
from transformers import PretrainedConfig

class McoreToHFWeightConverterBase:
    def __init__(self, hf_config: PretrainedConfig, mcore_config: TransformerConfig):
        self.hf_config = hf_config
        self.mcore_config = mcore_config

    def convert_param(self, name: str, params_one_group: list[torch.Tensor]) -> torch.Tensor:
        raise NotImplementedError

class McoreToHFWeightConverterDense(McoreToHFWeightConverterBase):
    def _convert_attention_param(self, name: str, params: list[torch.Tensor]) -> tuple[list[str], list[torch.Tensor]]:

        layer_number = name.split(".")[2]
        convert_names = []
        if "self_attention.linear_qkv.bias" in name or "self_attention.linear_qkv.weight" in name:
            param_type = name.split(".")[-1]
            assert param_type == "bias" or param_type == "weight"
            convert_names.append(f"model.layers.{layer_number}.self_attn.q_proj.{param_type}")
            convert_names.append(f"model.layers.{layer_number}.self_attn.k_proj.{param_type}")
            convert_names.append(f"model.layers.{layer_number}.self_attn.v_proj.{param_type}")
            assert len(params) == 3
        elif "self_attention.linear_proj.weight" in name:
            convert_names.append(f"model.layers.{layer_number}.self_attn.o_proj.weight")
            assert len(params) == 1
        elif "self_attention.linear_qkv.layer_norm_weight" in name:
            convert_names.append(f"model.layers.{layer_number}.input_layernorm.weight")
            assert len(params) == 1
        else:
            raise NotImplementedError(f"Unsupported parameter name: {name}")
        return convert_names, params

    def _convert_mlp_param(self, name: str, params: list[torch.Tensor]) -> tuple[list[str], list[torch.Tensor]]:

        layer_number = name.split(".")[2]
        convert_names = []
        if "mlp.linear_fc1.weight" in name:
            
            convert_names.append(f"model.layers.{layer_number}.mlp.gate_proj.weight")
            convert_names.append(f"model.layers.{layer_number}.mlp.up_proj.weight")
            assert len(params) == 2
        elif "mlp.linear_fc1.layer_norm_weight" in name:
            convert_names.append(f"model.layers.{layer_number}.post_attention_layernorm.weight")
            assert len(params) == 1
        elif "mlp.linear_fc2.weight" in name:
            convert_names.append(f"model.layers.{layer_number}.mlp.down_proj.weight")
            assert len(params) == 1
        else:
            raise NotImplementedError(f"Unsupported parameter name: {name}")
        return convert_names, params

    def convert_param(self, name: str, params_one_group: list[torch.Tensor]) -> tuple[list[str], list[torch.Tensor]]:
        direct_name_mapping = {
            "embedding.word_embeddings.weight": "model.embed_tokens.weight",
            "decoder.final_layernorm.weight": "model.norm.weight",
            "output_layer.weight": "lm_head.weight",
        }
        if name in direct_name_mapping:
            return [direct_name_mapping[name]], [params_one_group[0]]

        if "self_attention" in name:
            return self._convert_attention_param(name, params_one_group)
        elif "mlp" in name:
            return self._convert_mlp_param(name, params_one_group)
        else:
            raise NotImplementedError(f"Unsupported parameter name: {name}")

class McoreToHFWeightConverterQwen2Moe(McoreToHFWeightConverterDense):
    def _convert_mlp_param(self, name: str, params: list[torch.Tensor]) -> tuple[list[str], list[torch.Tensor]]:

        layer_number = name.split(".")[2]
        convert_names = []
        if "pre_mlp_layernorm" in name:
            convert_names.append(f"model.layers.{layer_number}.post_attention_layernorm.weight")
            assert len(params) == 1
        elif "mlp.router.weight" in name:
            convert_names.append(f"model.layers.{layer_number}.mlp.gate.weight")
            assert len(params) == 1
        elif "shared_experts.gate_weight" in name:
            convert_names.append(f"model.layers.{layer_number}.mlp.shared_expert_gate.weight")
            assert len(params) == 1
        elif "shared_experts.linear_fc1.weight" in name:  
            convert_names.append(f"model.layers.{layer_number}.mlp.shared_expert.gate_proj.weight")
            convert_names.append(f"model.layers.{layer_number}.mlp.shared_expert.up_proj.weight")
            assert len(params) == 2
        elif "shared_experts.linear_fc2.weight" in name:
            convert_names.append(f"model.layers.{layer_number}.mlp.shared_expert.down_proj.weight")
            assert len(params) == 1
        elif "mlp.experts.linear_fc1" in name:  
            expert_id = name.split("weight")[-1]
            convert_names.append(f"model.layers.{layer_number}.mlp.experts.{expert_id}.gate_proj.weight")
            convert_names.append(f"model.layers.{layer_number}.mlp.experts.{expert_id}.up_proj.weight")
            assert len(params) == 2
        elif "mlp.experts.linear_fc2" in name:
            expert_id = name.split("weight")[-1]
            convert_names.append(f"model.layers.{layer_number}.mlp.experts.{expert_id}.down_proj.weight")
            assert len(params) == 1
        else:
            raise NotImplementedError(f"Unsupported parameter name: {name}")
        return convert_names, params

class McoreToHFWeightConverterMixtral(McoreToHFWeightConverterDense):
    def _convert_mlp_param(self, name: str, params: list[torch.Tensor]) -> tuple[list[str], list[torch.Tensor]]:

        layer_number = name.split(".")[2]
        convert_names = []
        if "pre_mlp_layernorm" in name:
            convert_names.append(f"model.layers.{layer_number}.post_attention_layernorm.weight")
        elif "mlp.router.weight" in name:
            convert_names.append(f"model.layers.{layer_number}.block_sparse_moe.gate.weight")
        elif "mlp.experts.linear_fc1.weight" in name:
            expert_id = name.split("weight")[-1]
            convert_names.append(f"model.layers.{layer_number}.block_sparse_moe.experts.{expert_id}.w1.weight")
            convert_names.append(f"model.layers.{layer_number}.block_sparse_moe.experts.{expert_id}.w3.weight")
        elif "mlp.experts.linear_fc2.weight" in name:
            expert_id = name.split("weight")[-1]
            convert_names.append(f"model.layers.{layer_number}.block_sparse_moe.experts.{expert_id}.w2.weight")
        else:
            raise NotImplementedError(f"Unsupported parameter name: {name}")
        return convert_names, params
