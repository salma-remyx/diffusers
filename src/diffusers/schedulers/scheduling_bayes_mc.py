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

import torch

from ..configuration_utils import register_to_config
from .scheduling_discrete_ddim import DiscreteDDIMScheduler


class BayesMCScheduler(DiscreteDDIMScheduler):
    """
    Ancestral discrete DDIM sampling with Monte Carlo marginalization over the forward corruption, following
    "Bayesian Discrete Diffusion Beats Autoregressive Perplexity" (https://huggingface.co/papers/2507.07586).

    The paper shows that the expected denoiser output under the forward masking distribution recovers the exact
    posterior over clean tokens, so averaging the denoiser over `ensemble_size` independent corruptions of the
    current sequence estimates that posterior at rate O(1/sqrt(K)) with no extra training. This scheduler exposes
    that estimator for the uniform corruption process: [`~BayesMCScheduler.renoise_ensemble`] draws the K
    independent corrupted views, the caller runs the denoiser once per view, and
    [`~BayesMCScheduler.marginalize`] averages the per-view token probabilities at the corrupted positions into a
    single posterior estimate that the inherited [`~DiscreteDDIMScheduler.step`] samples from.

    Each denoising step costs `ensemble_size` denoiser forwards instead of one; `ensemble_size=1` recovers plain
    [`~DiscreteDDIMScheduler`] sampling. The ensemble replaces the predictor-corrector of
    [`~DiscreteDDIMScheduler`]: both refine the same posterior, so this scheduler does not register the corrector
    config keys and the inherited [`~DiscreteDDIMScheduler.step_correct`] is not meant to be driven on it.

    Args:
        num_inference_steps (`int`, defaults to 32):
            The number of denoising steps, defining the linear time grid the posterior is evaluated on.
        ensemble_size (`int`, defaults to 4):
            Number of independent corruptions marginalized per step (`K` in the paper).
    """

    order = 1

    @register_to_config
    def __init__(
        self,
        num_inference_steps: int = 32,
        ensemble_size: int = 4,
    ):
        if ensemble_size <= 0:
            raise ValueError(f"`ensemble_size` must be > 0, got {ensemble_size}.")
        # Same state as `DiscreteDDIMScheduler.__init__`, set directly so the config surface stays the two
        # ensemble-relevant keys instead of the parent's corrector knobs.
        self.num_inference_steps = num_inference_steps
        self.timesteps = torch.arange(num_inference_steps, dtype=torch.long)
        self.ensemble_size = ensemble_size

    def renoise_ensemble(
        self,
        sample: torch.LongTensor,
        timestep: int | torch.Tensor,
        vocab_size: int,
        *,
        generator: torch.Generator | None = None,
    ) -> tuple[torch.LongTensor, torch.BoolTensor]:
        """
        Draw `ensemble_size` independent corrupted views of `sample` at the current noise level.

        Each view replaces every position with a uniformly random token with probability `1 - alpha_t`, the uniform
        corruption rate at time `t`, mirroring one draw of the forward process the posterior marginalizes over.

        Args:
            sample (`torch.LongTensor` of shape `(batch_size, block_length)`):
                Current block token IDs `x_t`.
            timestep (`int` or `torch.Tensor`):
                Current step index within the denoising schedule, in `[0, num_inference_steps - 1]`.
            vocab_size (`int`):
                Number of tokens to draw the random replacements from.
            generator (`torch.Generator`, *optional*):
                RNG for sampling.

        Returns:
            `tuple[torch.LongTensor, torch.BoolTensor]`: the corrupted views of shape
            `(ensemble_size, batch_size, block_length)` and the boolean mask of corrupted positions of the same
            shape, where `True` marks a position that was re-corrupted in that view.
        """
        step_index = timestep.item() if isinstance(timestep, torch.Tensor) else timestep
        corrupt_rate = 1.0 - self._alpha(step_index)

        views = sample.unsqueeze(0).repeat(self.ensemble_size, 1, 1)
        corrupted = torch.rand(views.shape, device=sample.device, generator=generator) < corrupt_rate
        random_tokens = torch.randint(
            low=0, high=vocab_size, size=views.shape, device=sample.device, generator=generator
        )
        return torch.where(corrupted, random_tokens, views), corrupted

    def marginalize(
        self, ensemble_logits: torch.Tensor, corrupted: torch.BoolTensor
    ) -> torch.Tensor:
        """
        Average the per-view token probabilities into a Monte Carlo posterior estimate.

        Following Algorithm 1 of https://huggingface.co/papers/2507.07586, only the views that corrupted a position
        contribute to that position's average, since those are the draws of the forward process the posterior
        marginalizes over. A position corrupted in no view keeps the plain average over all views, where the
        denoiser reads it uncorrupted.

        Args:
            ensemble_logits (`torch.Tensor` of shape `(ensemble_size, batch_size, block_length, vocab_size)`):
                Denoiser logits, one tensor per corrupted view from [`~BayesMCScheduler.renoise_ensemble`].
            corrupted (`torch.BoolTensor` of shape `(ensemble_size, batch_size, block_length)`):
                The corrupted-positions mask returned by [`~BayesMCScheduler.renoise_ensemble`].

        Returns:
            `torch.Tensor` of shape `(batch_size, block_length, vocab_size)`: the log of the averaged token
            probabilities, so the inherited `step` samples directly from the posterior estimate.
        """
        probs = torch.softmax(ensemble_logits.float(), dim=-1)
        mask = corrupted.unsqueeze(-1).to(probs.dtype)

        summed = (probs * mask).sum(dim=0)
        counts = mask.sum(dim=0)
        averaged = torch.where(counts > 0, summed / counts.clamp(min=1.0), probs.mean(dim=0))
        return torch.log(averaged.clamp_min(torch.finfo(averaged.dtype).tiny))


__all__ = ["BayesMCScheduler"]
