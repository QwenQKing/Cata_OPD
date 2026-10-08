pass

import logging
import os
from functools import partial
from typing import Iterable

import torch
import torch.distributed
from megatron.core import parallel_state as mpu
from megatron.core.optimizer import DistributedOptimizer, OptimizerConfig
from megatron.core.pipeline_parallel import get_forward_backward_func
from omegaconf import OmegaConf
from torch import nn

from verl import DataProto
from verl.trainer.ppo import core_algos
from verl.utils.debug import GPUMemoryLogger
from verl.utils.megatron.pipeline_parallel import make_batch_generator
from verl.utils.py_functional import append_to_dict
from verl.utils.torch_functional import broadcast_dict_tensor, masked_mean, split_dict_tensor_into_batches
from verl.workers.critic import BasePPOCritic

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

class MegatronPPOCritic(BasePPOCritic):
    def __init__(
        self,
        config,
        model_config,
        hf_config,
        tf_config,
        critic_module: nn.ModuleList,
        critic_optimizer: DistributedOptimizer,
        critic_optimizer_config: OptimizerConfig,
    ):
        super().__init__(config=config)
        self._validate_config(config)
        self.model_config = model_config
        self.hf_config = hf_config  
        self.tf_config = tf_config  

        self.critic_module = critic_module
        self.critic_optimizer = critic_optimizer
        self.critic_optimizer_config = critic_optimizer_config

        self.optimizer_step_args = OmegaConf.create(
            {
                "skip_grad": None,
                "overlap_dp_param_comm": False,
                "overlap_dp_grad_comm": False,
                "gradient_accumulation_steps": 1,
                "sequence_parallel": self.tf_config.sequence_parallel,
                "DDP_impl": "local",
                "layernorm_allreduce_bucket_threshold": 0,
                "pipeline_model_parallel_split_rank": None,
                "reduce_grads_use_alltoall": False,
            }
        )

    def _validate_config(self, config) -> None:
        pass
        assert config.get("ulysses_sequence_parallel_size", 1) == 1
        if config.shuffle:
            assert config.data_loader_seed is not None, "If shuffle dataloader, seed must be manually set"
        if config.megatron.tensor_model_parallel_size == 1:
            print("[Warining] Because critic tp size == 1, set sp to False")
            config.megatron.sequence_parallel = False
        self.config = config

    @GPUMemoryLogger("megatron critic", logger=logger)
    def compute_values(self, data: DataProto) -> DataProto:
        
        responses = data.batch["responses"]
        attention_mask = data.batch["attention_mask"]
        response_length = responses.size(1)
        with torch.no_grad():
            output = self.forward_backward_batch(data=data, forward_only=True)
            if mpu.is_pipeline_last_stage(ignore_virtual=True):
                
                values = torch.cat([o["vpreds"] for o in output], dim=0)  
                values = values.to(torch.float32)
            else:
                values = torch.empty_like(attention_mask, dtype=torch.float32)

            values = values * attention_mask
            values = values[:, -response_length - 1 : -1]
            values = values.contiguous()

            torch.distributed.broadcast(
                tensor=values,
                src=mpu.get_pipeline_model_parallel_last_rank(),
                group=mpu.get_pipeline_model_parallel_group(),
            )

        torch.cuda.empty_cache()

        return values

    def make_minibatch_iterator(self, data: DataProto) -> Iterable[DataProto]:
        select_keys = ["input_ids", "responses", "attention_mask", "position_ids", "values", "returns"]
        data = data.select(batch_keys=select_keys)
        return data.make_iterator(
            mini_batch_size=self.config.ppo_mini_batch_size,
            epochs=self.config.ppo_epochs,
            seed=self.config.data_loader_seed,
            dataloader_kwargs={"shuffle": self.config.shuffle},
        )

    def forward_backward_batch(self, data: DataProto, forward_only=False):
        
        data.batch = data.batch.contiguous()
        broadcast_dict_tensor(data.batch, src=mpu.get_pipeline_model_parallel_last_rank(), group=mpu.get_pipeline_model_parallel_group())
        
        data.batch["attention_mask"] = data.batch["attention_mask"].to(bool)
        batches = split_dict_tensor_into_batches(data.batch, batch_size=self.config.ppo_micro_batch_size_per_gpu)
        n_micro_batch = len(batches)
        seq_len = batches[0]["input_ids"].shape[1]

        forward_backward_func = get_forward_backward_func()

        def loss_func(output, data, meta_info):
            if forward_only:
                return torch.tensor(1.0, device=output.device), {"vpreds": output}

            responses = data["responses"]
            attention_mask = data["attention_mask"]
            values = data["values"]
            returns = data["returns"]
            response_length = responses.size(1)

            response_mask = attention_mask[:, -response_length:]

            cliprange_value = self.config.cliprange_value

            vpreds = output  
            vpreds = vpreds[:, -response_length - 1 : -1]

            vf_loss, vf_clipfrac = core_algos.compute_value_loss(
                vpreds=vpreds,
                values=values,
                returns=returns,
                response_mask=response_mask,
                cliprange_value=cliprange_value,
            )
            stats = {
                "critic/vf_loss": vf_loss.detach().item(),
                "critic/vf_clipfrac": vf_clipfrac.detach().item(),
                "critic/vpred_mean": masked_mean(vpreds, response_mask).detach().item(),
            }

            return vf_loss, stats

        def forward_step(batch_iter, model):
            batch = next(batch_iter)
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]
            position_ids = batch["position_ids"]
            from verl.models.mcore import get_mcore_forward_fn

            forward_fn = get_mcore_forward_fn(self.hf_config)

            output = forward_fn(
                model,
                input_ids,
                attention_mask,
                position_ids,
                sequence_parallel=self.tf_config.sequence_parallel,
                value_model=True,
            )

            return output, partial(loss_func, data=batch, meta_info={})

        batch_generator = make_batch_generator(batches, vpp_size=len(self.critic_module))

        if mpu.get_pipeline_model_parallel_world_size() > 1:
            losses_reduced = forward_backward_func(
                forward_step_func=forward_step,
                data_iterator=batch_generator,
                model=self.critic_module,
                num_microbatches=n_micro_batch,
                seq_length=self.config.ppo_micro_batch_size_per_gpu * seq_len,  
                micro_batch_size=1,  
                forward_only=forward_only,
            )
        else:
            losses_reduced = forward_backward_func(
                forward_step_func=forward_step,
                data_iterator=batch_generator,
                model=self.critic_module,
                num_microbatches=n_micro_batch,
                seq_length=self.config.ppo_micro_batch_size_per_gpu * seq_len,  
                micro_batch_size=1,  
                forward_only=forward_only,
            )
        
        return losses_reduced

    @GPUMemoryLogger("megatron critic", logger=logger)
    def update_critic(self, dataloader: Iterable[DataProto]):
        metrics = {}

        for data in dataloader:
            
            self.critic_optimizer.zero_grad()
            
            for chunk in self.critic_module:
                chunk.zero_grad_buffer()

            metric_micro_batch = self.forward_backward_batch(data)

            update_successful, grad_norm, num_zeros_in_grad = self.critic_optimizer.step()

            if update_successful:
                
                pass
            else:
                raise NotImplementedError

            for metric in metric_micro_batch:
                append_to_dict(metrics, metric)  

        torch.cuda.empty_cache()
        return metrics
