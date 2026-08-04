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

"""Inference-time Sway Sampling for flow-matching schedulers.

Sway Sampling is an inference-time resampling of the flow steps introduced by F5-TTS
(Chen et al., 2024, "F5-TTS: A Fairytaler that Fakes Fluent and Faithful Speech with
Flow Matching", https://arxiv.org/abs/2410.06885). It redistributes the denoising
timesteps at inference time without retraining, so it can be applied to any pretrained
flow-matching model.

The reference transform (``SWivid/F5-TTS`` ``src/f5_tts/model/cfm.py``), applied to the
ascending flow fraction ``t`` (``t = 0`` is pure noise, ``t = 1`` is clean data), is::

    t = t + sway_coef * (cos(pi / 2 * t) - 1 + t)

with the F5-TTS default ``sway_coef = -1``. It is mapped here to the diffusers sigma
convention (``sigma = 1 - t``: ``sigma = 1`` is the pure-noise start of denoising), so the
returned schedule is descending ``1 -> 0`` as Euler flow-matching schedulers expect. With
``sway_coef = -1`` more steps land near ``sigma = 1`` (the noisy start of denoising);
``sway_coef = 0`` recovers the uniform schedule.
"""

import math

import torch


def sway_sampling(num_inference_steps: int, sway_coef: float = -1.0) -> torch.Tensor:
    r"""
    F5-TTS Sway Sampling schedule for flow-matching schedulers.

    Returns ``num_inference_steps`` flow steps (noise levels) in ``[0, 1]`` in descending
    order (``1 -> 0``), as expected by Euler flow-matching schedulers.

    This is an inference-time resampling of the flow steps and requires no retraining.
    With the F5-TTS default ``sway_coef = -1`` more steps are placed near ``sigma = 1``
    (the noisy start of denoising); ``sway_coef = 0`` is the uniform (linear) schedule.

    Args:
        num_inference_steps (`int`):
            Number of denoising steps the schedule should cover.
        sway_coef (`float`, defaults to `-1.0`):
            Sway sampling coefficient. ``-1`` matches F5-TTS; ``0`` recovers the uniform
            schedule.

    Returns:
        `torch.Tensor` of shape `(num_inference_steps,)`, descending sigma values in `[0, 1]`.
    """
    t = torch.linspace(0, 1, num_inference_steps, dtype=torch.float32)
    # Exact F5-TTS transform (SWivid/F5-TTS, cfm.py): biases the flow-step grid.
    t = t + sway_coef * (torch.cos(math.pi / 2 * t) - 1 + t)
    # Map the F5-TTS flow fraction (t = 0 is noise) to the diffusers sigma convention
    # (sigma = 1 is the pure-noise start); 1 - t is descending 1 -> 0.
    return 1.0 - t


def apply_sway_sampling(
    scheduler,
    num_inference_steps: int,
    sway_coef: float = -1.0,
    device: str | torch.device | None = None,
):
    r"""
    Replace a flow-matching Euler scheduler's sigma schedule with the F5-TTS Sway Sampling
    schedule.

    The sway schedule (see :func:`sway_sampling`) is rescaled to the scheduler's own
    ``[sigma_min, sigma_max]`` range and forwarded to the scheduler's ``set_timesteps``
    through its ``sigmas`` argument, so device placement, timestep bookkeeping and the
    appended terminal sigma are handled exactly as for a normal schedule. Call it in place
    of a plain ``scheduler.set_timesteps(num_inference_steps)``.

    Example:

        from diffusers import FlowMatchEulerDiscreteScheduler
        from diffusers.schedulers.sway_sampling import apply_sway_sampling

        scheduler = FlowMatchEulerDiscreteScheduler(shift=1.0)
        apply_sway_sampling(scheduler, num_inference_steps=30, sway_coef=-1.0)

    The scheduler's own timestep ``shift`` composes on top of the sway schedule, so set
    ``shift = 1.0`` for an unmodified F5-TTS schedule.

    Args:
        scheduler:
            A flow-matching scheduler exposing
            ``set_timesteps(num_inference_steps, sigmas=, device=)`` and ``sigma_min`` /
            ``sigma_max`` attributes, e.g. `FlowMatchEulerDiscreteScheduler`.
        num_inference_steps (`int`):
            Number of denoising steps.
        sway_coef (`float`, defaults to `-1.0`):
            Sway sampling coefficient; ``-1`` matches F5-TTS, ``0`` is uniform.
        device (`str` or `torch.device`, *optional*):
            Device to place the schedule on; forwarded to ``set_timesteps``.

    Returns:
        The same `scheduler` instance, with sway-sampled `sigmas` and `timesteps`.
    """
    sigmas = sway_sampling(num_inference_steps, sway_coef)
    sigmas = scheduler.sigma_min + sigmas * (scheduler.sigma_max - scheduler.sigma_min)
    scheduler.set_timesteps(num_inference_steps, sigmas=sigmas.tolist(), device=device)
    return scheduler
