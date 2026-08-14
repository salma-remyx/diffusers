# Copyright 2026 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
# the License. You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
# an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from ..models.attention_processor import Attention
from ..utils import logging
from .hooks import HookRegistry, ModelHook


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


_LOSA_SPARSE_ATTENTION_HOOK = "losa_sparse_attention"


@dataclass
class LoSAConfig:
    r"""
    Configuration for [LoSA: Near-Lossless Sparse Attention for Training-Free Video Diffusion Acceleration]
    (https://arxiv.org/abs/2608.12032v1).

    LoSA fixes a *retained attention mass* threshold rather than a sparsity ratio: at one early denoising step it
    measures the exact per-head, per-query-block key/value block masses, keeps the smallest block set that reaches the
    threshold for each query block, and reuses those frozen block indices for every remaining step.

    Args:
        retained_mass_threshold (`float`, defaults to `0.99`):
            The fraction of attention mass that must be retained for each (head, query block). The closer to 1.0, the
            higher the fidelity and the lower the sparsity.
        block_size (`int`, defaults to `64`):
            The number of tokens per key/value block. The sequence length need not be a multiple of `block_size`; the
            final (partial) block is handled exactly.
        profile_step (`int`, defaults to `0`):
            The forward-pass iteration (counted from the last `reset_stateful_hooks` call) at which the dense profiling
            pass runs and the block indices are frozen. Steps before this one run dense.
    """

    retained_mass_threshold: float = 0.99
    block_size: int = 64
    profile_step: int = 0

    def __post_init__(self):
        if not 0.0 < self.retained_mass_threshold <= 1.0:
            raise ValueError("`retained_mass_threshold` must be in the half-open interval (0.0, 1.0].")
        if self.block_size <= 0:
            raise ValueError("`block_size` must be a positive integer.")
        if self.profile_step < 0:
            raise ValueError("`profile_step` must be a non-negative integer.")

    def __repr__(self) -> str:
        return (
            f"LoSAConfig(\n"
            f"  retained_mass_threshold={self.retained_mass_threshold},\n"
            f"  block_size={self.block_size},\n"
            f"  profile_step={self.profile_step}\n"
            ")"
        )


class LoSAState:
    r"""
    State for LoSA sparse attention.

    Attributes:
        iteration (`int`):
            Number of forward passes run since the last reset. Used to decide whether the current pass is the profiling
            pass.
        profiled (`bool`):
            Whether the block indices have been frozen.
        selected (`list[tuple[torch.Tensor, torch.Tensor]] | None`):
            For each query block, a `(positions, validity)` tuple. `positions` is a `(batch * heads, capacity)` long
            tensor of selected key indices (padded), and `validity` is the matching boolean mask.
    """

    def __init__(self) -> None:
        self.iteration = 0
        self.profiled = False
        self.selected = None

    def reset(self):
        self.iteration = 0
        self.profiled = False
        self.selected = None


class LoSAHook(ModelHook):
    r"""A hook that applies LoSA near-lossless sparse attention to a self-attention layer."""

    _is_stateful = True

    def __init__(self, config: LoSAConfig) -> None:
        super().__init__()
        self.config = config

    def initialize_hook(self, module):
        self.state = LoSAState()
        return module

    def reset_state(self, module: torch.nn.Module) -> None:
        self.state.reset()
        return module

    def new_forward(
        self,
        module: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        # LoSA targets the long self-attention sequence of video diffusion transformers. Cross-attention (short,
        # fixed encoder sequence), masked attention, 4D convolutional inputs, grouped/added-kv projections and grouped
        # query attention fall outside that target and are left dense.
        head_dim = module.inner_dim // module.heads
        norm_ok = (module.norm_q is None or _norm_matches_head_dim(module.norm_q, head_dim)) and (
            module.norm_k is None or _norm_matches_head_dim(module.norm_k, head_dim)
        )
        is_supported_self_attention = (
            encoder_hidden_states is None
            and attention_mask is None
            and hidden_states.ndim == 3
            and module.scale_qk
            and module.group_norm is None
            and module.spatial_norm is None
            and module.to_k is not None
            and module.to_v is not None
            and module.to_out is not None
            and module.added_kv_proj_dim is None
            and module.inner_kv_dim == module.inner_dim
            and norm_ok
        )
        if not is_supported_self_attention or self.state.iteration < self.config.profile_step:
            return self.fn_ref.original_forward(
                hidden_states, encoder_hidden_states=encoder_hidden_states, attention_mask=attention_mask, **kwargs
            )

        residual = hidden_states
        batch_size, sequence_length, _ = hidden_states.shape
        heads = module.heads

        query = module.to_q(hidden_states).view(batch_size, sequence_length, heads, head_dim).transpose(1, 2)
        key = module.to_k(hidden_states).view(batch_size, sequence_length, heads, head_dim).transpose(1, 2)
        value = module.to_v(hidden_states).view(batch_size, sequence_length, heads, head_dim).transpose(1, 2)
        # `norm_ok` (checked above) guarantees any present norm normalizes over `head_dim`, matching the processor.
        if module.norm_q is not None:
            query = module.norm_q(query)
        if module.norm_k is not None:
            key = module.norm_k(key)

        # (batch, heads, seq, head_dim) -> (batch * heads, seq, head_dim) for per-head block selection.
        query_bh = query.reshape(batch_size * heads, sequence_length, head_dim)
        key_bh = key.reshape(batch_size * heads, sequence_length, head_dim)
        value_bh = value.reshape(batch_size * heads, sequence_length, head_dim)

        if not self.state.profiled:
            hidden = self._profile_and_attend(query_bh, key_bh, value_bh)
            self.state.profiled = True
        else:
            hidden = _attend_sparse(query_bh, key_bh, value_bh, self.state.selected, self.config.block_size)

        hidden = hidden.reshape(batch_size, heads, sequence_length, head_dim).transpose(1, 2).reshape(
            batch_size, sequence_length, heads * head_dim
        )
        hidden = hidden.to(query.dtype)

        hidden = module.to_out[0](hidden)
        hidden = module.to_out[1](hidden)
        if module.residual_connection:
            hidden = hidden + residual
        hidden = hidden / module.rescale_output_factor

        self.state.iteration += 1
        return hidden

    def _profile_and_attend(
        self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor
    ) -> torch.Tensor:
        r"""
        Run one dense pass to measure per (head, query block) key/value block masses, freeze the smallest block set
        reaching `retained_mass_threshold` for each query block, and return the dense output of this pass.
        """
        block_size = self.config.block_size
        threshold = self.config.retained_mass_threshold
        batch_heads, sequence_length, head_dim = query.shape
        num_blocks = (sequence_length + block_size - 1) // block_size
        padding = num_blocks * block_size - sequence_length

        selected = []
        for block_index in range(num_blocks):
            start, end = block_index * block_size, min((block_index + 1) * block_size, sequence_length)
            query_block = query[:, start:end, :]
            # Exact attention masses of this query block against every key, reduced to key blocks. The padding (zeros)
            # fills out the partial final block without contributing mass.
            scores = torch.bmm(query_block, key.transpose(-1, -2)) * (head_dim**-0.5)
            probs = torch.softmax(scores, dim=-1).sum(dim=1)  # (batch_heads, sequence_length)
            probs = F.pad(probs, (0, padding)) if padding else probs
            block_mass = probs.reshape(batch_heads, num_blocks, block_size).sum(dim=-1)

            total_mass = block_mass.sum(dim=-1, keepdim=True).clamp_min(torch.finfo(block_mass.dtype).tiny)
            fraction = block_mass / total_mass
            sorted_fraction, order = torch.sort(fraction, dim=-1, descending=True)
            cumulative = torch.cumsum(sorted_fraction, dim=-1)
            reached = cumulative >= threshold  # last block completes the retained mass, so a True is always present
            per_head_needed = torch.where(
                reached.any(dim=-1),
                reached.float().argmax(dim=-1) + 1,
                torch.full((batch_heads,), num_blocks, device=query.device, dtype=torch.long),
            )
            max_needed = int(per_head_needed.max().item())

            chosen_blocks = order[:, :max_needed]  # (batch_heads, max_needed) key-block ids, padded
            rank_valid = torch.arange(max_needed, device=query.device) < per_head_needed.unsqueeze(-1)
            offsets = torch.arange(block_size, device=query.device)
            positions = chosen_blocks.unsqueeze(-1) * block_size + offsets  # (batch_heads, max_needed, block_size)
            validity = rank_valid.unsqueeze(-1) & (positions < sequence_length)
            positions = positions.clamp(max=sequence_length - 1).reshape(batch_heads, max_needed * block_size)
            validity = validity.reshape(batch_heads, max_needed * block_size)
            selected.append((positions, validity))

        self.state.selected = selected
        # This profiling pass is dense by construction, so its output is the near-lossless reference.
        return F.scaled_dot_product_attention(query, key, value)


def _norm_matches_head_dim(norm: torch.nn.Module, head_dim: int) -> bool:
    normalized_shape = getattr(norm, "normalized_shape", None)
    if isinstance(normalized_shape, (list, tuple)) and len(normalized_shape) >= 1:
        return normalized_shape[0] == head_dim
    return True


def _attend_sparse(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    selected: list[tuple[torch.Tensor, torch.Tensor]],
    block_size: int,
) -> torch.Tensor:
    r"""
    Reconstruct attention using the frozen per-query-block key/value block sets. Each query block attends only to its
    selected keys (softmax renormalized over the kept support), which is the near-lossless sparse reconstruction.
    """
    batch_heads, sequence_length, head_dim = query.shape
    output = query.new_zeros(batch_heads, sequence_length, head_dim)

    for block_index, (positions, validity) in enumerate(selected):
        start, end = block_index * block_size, min((block_index + 1) * block_size, sequence_length)
        query_block = query[:, start:end, :]
        gathered_key = torch.gather(key, 1, positions.unsqueeze(-1).expand(-1, -1, head_dim))
        gathered_value = torch.gather(value, 1, positions.unsqueeze(-1).expand(-1, -1, head_dim))
        # `attn_mask=True` keeps a position; padded slots are masked to -inf and excluded from the softmax.
        attended = F.scaled_dot_product_attention(
            query_block, gathered_key, gathered_value, attn_mask=validity.unsqueeze(-2)
        )
        output[:, start:end, :] = attended

    return output


def apply_losa_sparse_attention(module: torch.nn.Module, config: LoSAConfig):
    r"""
    Apply [LoSA](https://arxiv.org/abs/2608.12032v1) near-lossless sparse attention to a given module.

    LoSA is a training-free accelerator for the self-attention in video diffusion transformers. It fixes a retained
    attention-mass threshold (rather than a sparsity ratio): at one early dense step it measures, for every head and
    query block, the exact key/value block masses and keeps the smallest block set meeting the threshold; those frozen
    indices are reused for all remaining steps, so each query block attends only to its retained keys.

    Cross-attention layers are skipped (their encoder sequence is short); self-attention layers whose configuration is
    not supported by the hook (grouped query attention, added key/value projections, masked or 4D inputs) are left
    untouched and fall back to their original dense attention.

    Args:
        module (`torch.nn.Module`):
            The module to apply LoSA to. Typically the denoiser of a diffusion pipeline (e.g. `pipe.transformer`).
        config (`LoSAConfig`):
            The configuration to use for LoSA.

    Example:

    ```python
    >>> import torch
    >>> from diffusers.hooks.losa_sparse_attention import LoSAConfig, apply_losa_sparse_attention

    >>> # `pipe` is a video diffusion pipeline with a transformer denoiser.
    >>> config = LoSAConfig(retained_mass_threshold=0.99, block_size=64, profile_step=0)
    >>> apply_losa_sparse_attention(pipe.transformer, config)
    ```
    """
    for name, submodule in module.named_modules():
        if not isinstance(submodule, Attention):
            continue
        if getattr(submodule, "is_cross_attention", False):
            continue
        registry = HookRegistry.check_if_exists_or_initialize(submodule)
        hook = LoSAHook(config)
        registry.register_hook(hook, _LOSA_SPARSE_ATTENTION_HOOK)
        logger.debug(f"Enabling LoSA sparse attention in layer: {name}")
