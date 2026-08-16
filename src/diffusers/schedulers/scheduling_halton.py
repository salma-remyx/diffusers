# Copyright 2026 The HuggingFace Team. All rights reserved.
#
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

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from ..configuration_utils import ConfigMixin, register_to_config
from ..utils import BaseOutput
from .scheduling_utils import SchedulerMixin


@dataclass
class HaltonSchedulerOutput(BaseOutput):
    """
    Output class for the Halton scheduler.

    Args:
        prev_sample (`torch.LongTensor` of shape `(batch_size, block_length)`):
            Updated block tokens after the current denoising step.
        transfer_index (`torch.BoolTensor` of shape `(batch_size, block_length)`):
            Boolean mask indicating which tokens were committed in this step.
        sampled_tokens (`torch.LongTensor` of shape `(batch_size, block_length)`):
            Sampled token IDs from the model logits.
        sampled_probs (`torch.Tensor` of shape `(batch_size, block_length)`):
            Probabilities of the sampled tokens.
        pred_logits (`torch.Tensor` of shape `(batch_size, block_length, vocab_size)`):
            The temperature-scaled logits the candidates were drawn from, for self-conditioning the next step.
    """

    prev_sample: torch.LongTensor
    transfer_index: torch.BoolTensor
    sampled_tokens: torch.LongTensor
    sampled_probs: torch.Tensor
    pred_logits: torch.Tensor


class HaltonScheduler(SchedulerMixin, ConfigMixin):
    """
    Halton (low-discrepancy) position-selection scheduler for masked and uniform discrete diffusion.

    Instead of committing the tokens the model is most confident about, this scheduler commits the tokens at positions
    fixed in advance by a quasi-random Halton sequence, so the tokens revealed at every step are spread uniformly over
    the sequence instead of clustering on the easy positions. The paper shows the confidence rule commits mutually
    informative tokens together, and that decoupling *which* positions are committed from *what* the model predicts
    there reduces the non-recoverable sampling errors made during generation.

    The number of tokens committed at step `i` follows MaskGIT's arccos schedule (`1 - arccos(r) / (pi / 2)` for
    `r = (i + 1) / num_inference_steps`), floored at `i + 1` so every step commits at least one token. Positions are
    ranked by the first appearance of their cell in a 2D Halton sequence with the coprime bases (2, 3), so sequence
    order is the commit order.

    Proposed in "Halton Scheduler For Masked Generative Image Transformer"
    (https://huggingface.co/papers/2503.17076). Training-free: it is a drop-in replacement for any commit-and-renoise
    scheduler. Like [`EntropyBoundScheduler`], it anneals its own sampling temperature and ignores a per-call one.

    Args:
        num_inference_steps (`int`, defaults to 32):
            The maximum number of denoising steps.
        t_min (`float`, defaults to 1.0):
            Sampling temperature on the first denoising step.
        t_max (`float`, defaults to 1.0):
            Sampling temperature on the last denoising step.
        randomize (`bool`, defaults to `False`):
            Roll each sequence's Halton order by a random offset, so batch elements do not share one commit order.
    """

    order = 1

    @register_to_config
    def __init__(self, num_inference_steps: int = 32, t_min: float = 1.0, t_max: float = 1.0, randomize: bool = False):
        self.num_inference_steps = num_inference_steps
        self.timesteps = torch.arange(num_inference_steps, dtype=torch.long)
        # Commit order over the block positions, rebuilt by `set_timesteps` for the actual block length.
        self._ranks: torch.LongTensor | None = None
        self._committed: torch.BoolTensor | None = None

    def set_timesteps(self, num_inference_steps: int, device: str | torch.device | None = None) -> None:
        if num_inference_steps <= 0:
            raise ValueError(f"`num_inference_steps` must be > 0, got {num_inference_steps}.")
        self.num_inference_steps = num_inference_steps
        self.timesteps = torch.arange(num_inference_steps, device=device, dtype=torch.long)
        self._ranks = None
        self._committed = None

    @staticmethod
    def _halton_sequence(base: int, num_points: int) -> list[float]:
        """Halton radical-inverse values in `base`, in sequence order."""
        values = []
        n, d = 0, 1
        for _ in range(num_points):
            x = d - n
            if x == 1:
                n = 1
                d *= base
            else:
                y = d // base
                while x <= y:
                    y //= base
                n = (base + 1) * y - x
            values.append(n / d)
        return values

    @classmethod
    def build_position_ranks(cls, block_length: int) -> torch.LongTensor:
        """
        Rank the `block_length` positions by the first appearance of their cell in a 2D Halton sequence.

        The sequence points are scaled onto a `ceil(sqrt(block_length))`-sized grid and each cell keeps the index of the
        first point that landed in it; ranking cells by that index gives a low-discrepancy permutation of the
        positions. Positions whose cell is never hit rank last, so the result is always a full permutation.
        """
        if block_length <= 0:
            raise ValueError(f"`block_length` must be > 0, got {block_length}.")
        grid_size = math.isqrt(block_length - 1) + 1  # ceil(sqrt(block_length))
        num_points = 10 * block_length

        points = torch.tensor(
            list(zip(cls._halton_sequence(2, num_points), cls._halton_sequence(3, num_points))), dtype=torch.float64
        )
        cells = (points * grid_size).long().view(-1, 2)
        flat_cells = cells[:, 0] * grid_size + cells[:, 1]

        # First point index reaching each cell; `num_points` marks a cell no point reached.
        first_seen = torch.full((grid_size * grid_size,), num_points, dtype=torch.long)
        first_seen = torch.scatter_reduce(
            first_seen, 0, flat_cells, torch.arange(num_points), reduce="amin", include_self=True
        )

        # Cells outside the block rank after every reached cell, in position order.
        in_block = first_seen[:block_length]
        keys = torch.where(in_block < num_points, in_block, num_points + torch.arange(block_length))
        order = keys.argsort(stable=True)

        ranks = torch.empty(block_length, dtype=torch.long)
        ranks[order] = torch.arange(block_length)
        return ranks

    def _get_ranks(self, block_length: int, device: torch.device) -> torch.LongTensor:
        if self._ranks is None or self._ranks.numel() != block_length:
            self._ranks = self.build_position_ranks(block_length)
        return self._ranks.to(device=device)

    def step(
        self,
        model_output: torch.Tensor,
        timestep: int | torch.Tensor,
        sample: torch.LongTensor,
        *,
        mask_token_id: int | None = None,
        generator: torch.Generator | None = None,
        return_dict: bool = True,
    ) -> HaltonSchedulerOutput | tuple[torch.LongTensor, torch.BoolTensor, torch.LongTensor, torch.Tensor]:
        """
        Commit the next slice of Halton-ranked positions and renoise the rest.

        Args:
            model_output (`torch.Tensor` of shape `(batch_size, block_length, vocab_size)`):
                Raw logits from the model for the current block.
            timestep (`int` or `torch.Tensor`):
                Current step index within the denoising schedule.
            sample (`torch.LongTensor` of shape `(batch_size, block_length)`):
                Current block token IDs. In mask mode (`mask_token_id` passed) uncommitted positions hold the mask
                token; otherwise they are renoised with uniformly random tokens.
            mask_token_id (`int`, *optional*):
                Token ID used for masked positions; omit it to run the uniform-corruption (renoise) mode.
            generator (`torch.Generator`, *optional*):
                RNG for sampling tokens and for the `randomize` offsets.
            return_dict (`bool`):
                Whether to return a [`HaltonSchedulerOutput`] or a plain tuple.
        """
        if isinstance(timestep, torch.Tensor):
            step_index = int(timestep.item())
        else:
            step_index = int(timestep)

        batch_size, block_length = sample.shape
        ranks = self._get_ranks(block_length, sample.device).unsqueeze(0)
        if self.config.randomize:
            offsets = torch.randint(0, block_length, (batch_size, 1), device=sample.device, generator=generator)
            ranks = (ranks - offsets) % block_length
        # Commit quota: MaskGIT's arccos schedule, floored at one token per elapsed step.
        ratio = (step_index + 1) / self.num_inference_steps
        quota = max(step_index + 1, int((1 - math.acos(ratio) / (math.pi * 0.5)) * block_length))

        # Pipelines loop the step index back to 0 for each new canvas/block, so reset the committed state there, as
        # [`BlockRefinementScheduler`] does in its uniform-corruption mode.
        if step_index == 0 or self._committed is None or self._committed.shape != sample.shape:
            self._committed = torch.zeros_like(sample, dtype=torch.bool)
        committed = self._committed
        transfer_index = ~committed & (ranks < quota)

        # Anneal the temperature from `t_max` on the first step down to `t_min` on the last, and scale the logits once
        # so the returned self-conditioning logits match the distribution the candidates were drawn from.
        fraction = step_index / max(self.num_inference_steps - 1, 1)
        temperature = self.config.t_max + (self.config.t_min - self.config.t_max) * fraction
        scaled_logits = model_output / temperature

        vocab_size = model_output.shape[-1]
        probs = torch.softmax(scaled_logits.reshape(-1, vocab_size).float(), dim=-1)
        tokens = torch.multinomial(probs, num_samples=1, generator=generator)
        sampled_tokens = tokens.view(*model_output.shape[:-1])
        sampled_probs = torch.gather(probs, -1, tokens).view(*model_output.shape[:-1])

        self._committed = committed | transfer_index
        # Newly committed positions take their fresh sample; previously committed ones keep the token they hold.
        kept = torch.where(transfer_index, sampled_tokens, sample)
        if mask_token_id is None:
            rest = torch.randint(low=0, high=vocab_size, size=sample.shape, device=sample.device, generator=generator)
        else:
            rest = torch.full_like(sample, mask_token_id)
        prev_sample = torch.where(self._committed, kept, rest)

        if not return_dict:
            return prev_sample, transfer_index, sampled_tokens, sampled_probs, scaled_logits
        return HaltonSchedulerOutput(
            prev_sample=prev_sample,
            transfer_index=transfer_index,
            sampled_tokens=sampled_tokens,
            sampled_probs=sampled_probs,
            pred_logits=scaled_logits,
        )


__all__ = ["HaltonScheduler", "HaltonSchedulerOutput"]
