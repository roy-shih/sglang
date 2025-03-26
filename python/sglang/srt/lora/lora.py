# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

# Integrates "S-LoRA: Serving Thousands of Concurrent LoRA Adapters"
# and "Punica: Multi-Tenant LoRA Serving"

# LoRA layers class inheritance adapted from:
# https://github.com/vllm-project/vllm/blob/4abf6336ec65c270343eb895e7b18786e9274176/vllm/lora/layers.py

import re
from dataclasses import dataclass

import torch
from torch import nn

from sglang.srt.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from sglang.srt.layers.vocab_parallel_embedding import VocabParallelEmbedding
from sglang.srt.model_loader.loader import DefaultModelLoader


@dataclass
class LoraBatchInfo:
    # Batch size
    bs: int

    # Lengths of each sequence in shape (bs,)
    seg_lens: torch.Tensor

    # Indice pointers of each sequence in shape (bs + 1, )
    seg_indptr: torch.Tensor

    # Maximum sequence length of current batch
    max_len: int

    # The index of lora adapter used by each sequence, in shape (bs,)
    weight_indices: torch.Tensor


class BaseLayerWithLoRA(nn.Module):
    def __init__(self, base_layer, lora_rank, scaling, lora_backend):
        super().__init__()
        self.base_layer = base_layer
        self.lora_rank = lora_rank
        self.scaling = scaling
        self.set_lora = False
        self.lora_backend = lora_backend

    def forward(self, x: torch.Tensor):
        return self.base_layer.forward(x)

    def set_lora_info(self, *args):
        pass


class VocabParallelEmbeddingWithLoRA(BaseLayerWithLoRA):
    def __init__(
        self, base_layer: VocabParallelEmbedding, lora_rank, scaling, lora_backend
    ) -> None:
        super().__init__(base_layer, lora_rank, scaling, lora_backend)
        self.weight = base_layer.weight


class ColumnParallelLinearWithLoRA(BaseLayerWithLoRA):
    def __init__(
        self, base_layer: ColumnParallelLinear, lora_rank, scaling, lora_backend
    ) -> None:
        super().__init__(base_layer, lora_rank, scaling, lora_backend)

    def apply_lora(self, output: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        # TODO
        return output

    def forward(self, input_: torch.Tensor):
        # duplicate the logic in ColumnParallelLinear
        bias = self.base_layer.bias if not self.base_layer.skip_bias_add else None
        output_parallel = self.base_layer.quant_method.apply(
            self.base_layer, input_, bias
        )

        if self.set_lora:
            output_parallel = self.apply_lora(output_parallel, input_)

        if self.base_layer.gather_output:
            output = tensor_model_parallel_all_gather(output_parallel)
        else:
            output = output_parallel
        output_bias = self.base_layer.bias if self.base_layer.skip_bias_add else None
        return output, output_bias


class MergedColumnParallelLinearWithLoRA(ColumnParallelLinearWithLoRA):
    def __init__(
        self, base_layer: MergedColumnParallelLinear, lora_rank, scaling, lora_backend
    ) -> None:
        super().__init__(base_layer, lora_rank, scaling, lora_backend)

    def set_lora_info(
        self,
        A_buffer,
        B_buffer,
    ):
        self.set_lora = True
        self.A_buffer = A_buffer
        self.B_buffer = B_buffer

    def apply_lora(self, base_output: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        lora_a_output = self.lora_backend.run_lora_a_sgemm(x=x, weights=self.A_buffer)

        output_dim = base_output.shape[-1]
        lora_output = torch.empty_like(base_output)
        lora_output[:, :output_dim] = self.lora_backend.run_lora_b_sgemm(
            x=lora_a_output[:, 0 : self.lora_rank].contiguous(),
            weights=self.B_buffer[0],
        )

        lora_output[:, output_dim : 2 * output_dim] = (
            self.lora_backend.run_lora_b_sgemm(
                x=lora_a_output[:, self.lora_rank : 2 * self.lora_rank].contiguous(),
                weights=self.B_buffer[1],
            )
        )

        return base_output + lora_output * self.scaling


class QKVParallelLinearWithLoRA(ColumnParallelLinearWithLoRA):
    def init__(
        self, base_layer: QKVParallelLinear, lora_rank, scaling, lora_backend
    ) -> None:
        super().__init__(base_layer, lora_rank, scaling, lora_backend)

    def set_lora_info(
        self,
        A_buffer_qkv,
        B_buffer_q,
        B_buffer_kv,
    ):
        self.set_lora = True
        self.A_buffer_qkv = A_buffer_qkv

        if self.lora_backend.fuse_qkv_lora_b:
            assert (
                B_buffer_q.shape[-1] == B_buffer_kv.shape[-1]
            ), "The lora rank of q and kv should be the same when enabling fusion of qkv lora_b"
            output_dim_q, output_dim_kv = B_buffer_q.shape[-2], B_buffer_kv.shape[-2]

            # B_buffer_qkv: (num_lora, output_dim_q + 2 * output_dim_kv, r)
            self.B_buffer_qkv = torch.cat(
                (B_buffer_q[0], B_buffer_kv[0], B_buffer_kv[1]), dim=-2
            ).contiguous()

            # Offsets of q/k/v in output dimension
            self.output_offset = torch.tensor(
                [
                    0,
                    output_dim_q,
                    output_dim_q + output_dim_kv,
                    output_dim_q + 2 * output_dim_kv,
                ],
                dtype=torch.int32,
                device=B_buffer_q.device,
            )
            # For computing number of launched blocks
            self.max_qkv_out_dim = max(output_dim_q, output_dim_kv)
        else:
            self.B_buffer_qkv = (
                B_buffer_q,
                B_buffer_kv,
            )
            self.output_offset = None
            self.max_qkv_out_dim = None

    def apply_lora(self, base_output: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        lora_output = self.lora_backend.run_qkv_lora(
            x,
            self.A_buffer_qkv,
            self.B_buffer_qkv,
            output_offset=self.output_offset,
            max_qkv_out_dim=self.max_qkv_out_dim,
            base_output=base_output,
            scaling=self.scaling,
        )
        return (
            lora_output
            if self.lora_backend.fuse_output_scaling_add
            else base_output + lora_output * self.scaling
        )


class RowParallelLinearWithLoRA(BaseLayerWithLoRA):
    def __init__(
        self, base_layer: RowParallelLinear, lora_rank, scaling, lora_backend
    ) -> None:
        super().__init__(base_layer, lora_rank, scaling, lora_backend)

    def set_lora_info(self, A_buffer, B_buffer):
        self.set_lora = True
        self.A_buffer = A_buffer
        self.B_buffer = B_buffer

    def apply_lora(self, base_output: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        lora_a_output = self.lora_backend.run_lora_a_sgemm(x, self.A_buffer)
        lora_output = self.lora_backend.run_lora_b_sgemm(
            lora_a_output,
            self.B_buffer[0],
            base_output=base_output,
            scaling=self.scaling,
        )
        return (
            lora_output
            if self.lora_backend.fuse_output_scaling_add
            else base_output + lora_output * self.scaling
        )

    def forward(self, input_):
        # duplicate the logic in RowParallelLinear
        if self.base_layer.input_is_parallel:
            input_parallel = input_
        else:
            tp_rank = get_tensor_model_parallel_rank()
            splitted_input = split_tensor_along_last_dim(
                input_, num_partitions=self.base_layer.tp_size
            )
            input_parallel = splitted_input[tp_rank].contiguous()
        output_parallel = self.base_layer.quant_method.apply(
            self.base_layer, input_parallel
        )

        if self.set_lora:
            output_parallel = self.apply_lora(output_parallel, input_parallel)

        if self.base_layer.reduce_results and self.base_layer.tp_size > 1:
            output_ = tensor_model_parallel_all_reduce(output_parallel)
        else:
            output_ = output_parallel

        if not self.base_layer.skip_bias_add:
            output = (
                output_ + self.base_layer.bias
                if self.base_layer.bias is not None
                else output_
            )
            output_bias = None
        else:
            output = output_
            output_bias = self.base_layer.bias
        return output, output_bias


def get_lora_layer(
    layer: nn.Module, lora_rank, scaling, lora_backend
) -> BaseLayerWithLoRA:
    supported_layer_types = {
        # the order matters
        VocabParallelEmbedding: VocabParallelEmbeddingWithLoRA,
        QKVParallelLinear: QKVParallelLinearWithLoRA,
        MergedColumnParallelLinear: MergedColumnParallelLinearWithLoRA,
        ColumnParallelLinear: ColumnParallelLinearWithLoRA,
        RowParallelLinear: RowParallelLinearWithLoRA,
    }
    for src_layer_type, lora_layer_type in supported_layer_types.items():
        if isinstance(layer, src_layer_type):  # pylint: disable=unidiomatic-typecheck
            ret = lora_layer_type(layer, lora_rank, scaling, lora_backend)
            return ret
    raise Exception(f"No corresponding LoRA layer supported for {type(layer)}.")


def get_mapped_params(module_names):
    ret = set()
    for module_name in module_names:
        ret.add(params_mapping(module_name))
    return list(ret)


class LoRALayer(nn.Module):
    def __init__(self, config, base_hf_config):
        super().__init__()
        self.config = config
        self.base_hf_config = base_hf_config
        self.weights = {}
        self.weight_gpu = {}

    def load_to_gpu(self):
        for name, weight in self.weights.items():
            self.weight_gpu[name] = weight.to(torch.float16).to("cuda")

    def offload_from_gpu(self):
        for name, weight in self.weights.items():
            self.weight_gpu[name] = None


class LoRAAdapter(nn.Module):
    def __init__(self, uid, config, base_hf_config, load_config, lora_backend):
        super().__init__()
        self.uid = uid
        self.config = config
        assert self.config.hf_config["peft_type"].lower() == "lora"
        self.base_hf_config = base_hf_config
        self.load_config = load_config
        self.lora_backend = lora_backend
        self.scaling = self.config.lora_alpha / self.config.r

        self.layers = nn.ModuleList(
            [
                LoRALayer(config, base_hf_config)
                for i in range(base_hf_config.num_hidden_layers)
            ]
        )

        self.weights = {}
        self.weights_gpu = {}

    def get_stacked_multiply(self, module_name):
        stacked_rank = {
            "qkv_proj": 3,
            "kv_proj": 2,
            "gate_up_proj": 2,
        }
        return stacked_rank[module_name] if module_name in stacked_rank else 1

    def load_to_gpu(self):
        for name, weight in self.weights.items():
            self.weights_gpu[name] = weight.to(torch.float16).to("cuda")
        for layer in self.layers:
            layer.load_to_gpu()

    def offload_from_gpu(self):
        for name, weight in self.weights.items():
            self.weights_gpu[name] = None
        for layer in self.layers:
            layer.offload_from_gpu()

    # initialize the LoRA weights to cpu
    def initialize_weights(self):
        model_path = self.config.path
        loader = DefaultModelLoader(self.load_config)
        revision = getattr(self.config.hf_config, "revision", None)
        for name, loaded_weight in loader._get_weights_iterator(
            DefaultModelLoader.Source(
                model_path, revision=revision, fall_back_to_pt=True
            )
        ):
            match = re.search(r"layers\.(\d+)\.", name)
            if match is not None:
                layer_id = int(match.group(1))
                self.layers[layer_id].weights[name] = loaded_weight.cpu()
            else:
                self.weights[name] = loaded_weight.cpu()

        # stack kv_proj and gate_up_proj
        for i in range(self.base_hf_config.num_hidden_layers):
            layer = self.layers[i]
            weight_names = [name for name, _ in layer.weights.items()]
            for weight_name in weight_names:
                if "k_proj" in weight_name:
                    q_name = weight_name.replace("k_proj", "q_proj")
                    v_name = weight_name.replace("k_proj", "v_proj")
                    kv_name = weight_name.replace("k_proj", "kv_proj")
                    qkv_name = weight_name.replace("k_proj", "qkv_proj")
                    if "lora_A" in weight_name:
                        layer.weights[qkv_name] = torch.cat(
                            (
                                layer.weights[q_name],
                                layer.weights[weight_name],
                                layer.weights[v_name],
                            ),
                            0,
                        )
                        layer.weights.pop(q_name)
                        layer.weights.pop(weight_name)
                        layer.weights.pop(v_name)
                    else:
                        layer.weights[kv_name] = torch.stack(
                            [
                                layer.weights[weight_name],
                                layer.weights[v_name],
                            ],
                            dim=0,
                        )
                        layer.weights.pop(weight_name)
                        layer.weights.pop(v_name)
                elif "gate_proj" in weight_name:
                    up_name = weight_name.replace("gate_proj", "up_proj")
                    gate_up_name = weight_name.replace("gate_proj", "gate_up_proj")
                    if "lora_A" in weight_name:
                        layer.weights[gate_up_name] = torch.cat(
                            (layer.weights[weight_name], layer.weights[up_name]), 0
                        )
                    else:
                        layer.weights[gate_up_name] = torch.stack(
                            [layer.weights[weight_name], layer.weights[up_name]], dim=0
                        )
                    layer.weights.pop(weight_name)
                    layer.weights.pop(up_name)
