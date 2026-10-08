from abc import ABC, abstractmethod

from megatron.core.models.gpt.gpt_layer_specs import get_gpt_decoder_block_spec
from megatron.core.models.gpt.gpt_model import GPTModel

from .config_converter import PretrainedConfig, TransformerConfig

class BaseModelInitializer(ABC):
    pass

    def __init__(self, tfconfig: TransformerConfig, hf_config: PretrainedConfig):
        self.tfconfig = tfconfig
        self.hf_config = hf_config

    @abstractmethod
    def get_transformer_layer_spec(self):
        pass
        pass

    def get_rope_scaling_args(self) -> dict:
        pass
        rope_scaling_args = {}
        if "rope_scaling" in self.hf_config:
            if self.hf_config.rope_scaling is not None:
                assert self.hf_config.rope_scaling["type"] == "linear", "only linear scaling is supported for now"
                rope_scaling_args["seq_len_interpolation_factor"] = self.hf_config.rope_scaling["factor"]
        return rope_scaling_args

    def initialize(
        self,
        pre_process: bool = True,
        post_process: bool = True,
        share_embeddings_and_output_weights: bool = False,
        value: bool = False,
        **extra_kwargs,
    ) -> GPTModel:
        pass
        transformer_layer_spec = self.get_transformer_layer_spec()
        rope_scaling_args = self.get_rope_scaling_args()

        model = GPTModel(
            config=self.tfconfig,
            transformer_layer_spec=transformer_layer_spec,
            vocab_size=self.hf_config.vocab_size,
            max_sequence_length=self.hf_config.max_position_embeddings,
            pre_process=pre_process,
            post_process=post_process,
            share_embeddings_and_output_weights=share_embeddings_and_output_weights,
            position_embedding_type="rope",
            rotary_base=self.hf_config.rope_theta,
            **rope_scaling_args,
        )

        if post_process and value:
            from verl.models.llama.megatron.layers.parallel_linear import LinearForLastLayer

            model.output_layer = LinearForLastLayer(input_size=self.tfconfig.hidden_size, output_size=1, config=self.tfconfig)

        return model

class DenseModel(BaseModelInitializer):
    pass

    def get_transformer_layer_spec(self):
        assert self.tfconfig.normalization == "RMSNorm", "only RMSNorm is supported for now"
        return get_gpt_decoder_block_spec(self.tfconfig, use_transformer_engine=True)

class Qwen2MoEModel(BaseModelInitializer):
    pass

    def get_transformer_layer_spec(self):
        assert self.tfconfig.normalization == "RMSNorm", "only RMSNorm is supported for now"
        transformer_layer_spec = get_gpt_decoder_block_spec(self.tfconfig, use_transformer_engine=True)

        for i in range(len(transformer_layer_spec.layer_specs)):
            transformer_layer_spec.layer_specs[i].submodules.mlp.submodules.shared_experts.params["gate"] = True

        return transformer_layer_spec

    def initialize(self, freeze_moe_router: bool = True, **kwargs):
        
        model = super().initialize(**kwargs)
        if freeze_moe_router:
            for layer in model.decoder.layers:
                layer.mlp.router.weight.requires_grad = False
                layer.mlp.shared_experts.gate_weight.requires_grad = False
        return model

class MixtralModel(BaseModelInitializer):
    pass

    def get_transformer_layer_spec(self):
        assert self.tfconfig.normalization == "RMSNorm", "only RMSNorm is supported for now"
        transformer_layer_spec = get_gpt_decoder_block_spec(self.tfconfig, use_transformer_engine=True)
        return transformer_layer_spec

    def initialize(self, freeze_moe_router: bool = False, **kwargs):
        model = super().initialize(**kwargs)
        if freeze_moe_router:
            for layer in model.decoder.layers:
                layer.mlp.router.weight.requires_grad = False
        return model

class Qwen25VLModel(BaseModelInitializer):
    pass

    def get_transformer_layer_spec(self):
        raise NotImplementedError("VLM is not supported yet")
