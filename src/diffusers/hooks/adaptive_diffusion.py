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

from collections import deque
from dataclasses import dataclass

import torch

from ..utils import get_logger
from .hooks import BaseState, HookRegistry, ModelHook, StateManager


logger = get_logger(__name__)  # pylint: disable=invalid-name

_ADAPTIVE_DIFFUSION_HOOK = "adaptive_diffusion_hook"


@dataclass
class AdaptiveDiffusionConfig:
    r"""
    Configuration for
    [AdaptiveDiffusion](https://huggingface.co/papers/2410.09873), a training-free step cache that
    reuses the previous noise prediction when the input latent trajectory is predictable enough.

    Unlike the block-level caches in this package (First Block Cache, MagCache), AdaptiveDiffusion
    wraps the *whole* denoising model forward: it keeps the previous noise prediction and reuses it
    whenever the third-order finite difference of the input latent trajectory is bounded. The
    estimator is a function of the latents alone, so the skip decision costs no model evaluation.

    Args:
        threshold (`float`, defaults to `0.05`):
            Relative bound on the third-order latent difference. When
            `||x_t - 3 x_{t-1} + 3 x_{t-2} - x_{t-3}|| / ||x_t||` falls below this value, the
            previous noise prediction is reused and the model forward is skipped. A higher threshold
            skips more aggressively (faster inference) at the cost of fidelity.
        max_skip_steps (`int`, defaults to `3`):
            The maximum number of consecutive noise-prediction steps that may be skipped before a
            fresh prediction is forced. This bounds the error accumulated while reusing one cached
            prediction.
        num_inference_steps (`int`, defaults to `28`):
            The number of denoising steps in the pipeline. The step cache resets once this many
            forward calls have been seen, so a new generation starts from an empty cache.
    """

    threshold: float = 0.05
    max_skip_steps: int = 3
    num_inference_steps: int = 28


class AdaptiveDiffusionState(BaseState):
    def __init__(self) -> None:
        super().__init__()
        # Rolling window of recent input latents, newest last. A third-order finite difference needs
        # the current latent plus the three that precede it.
        self.latent_history: deque = deque(maxlen=4)
        # The most recently *computed* noise prediction, reused while a skip is taken.
        self.cached_output = None
        self.step_index: int = 0
        self.consecutive_skips: int = 0

    def reset(self):
        self.latent_history.clear()
        self.cached_output = None
        self.step_index = 0
        self.consecutive_skips = 0


class AdaptiveDiffusionHook(ModelHook):
    r"""
    A training-free step cache adapted from
    [AdaptiveDiffusion](https://huggingface.co/papers/2410.09873).

    The hook wraps a denoising model's forward (the per-step noise prediction) and reuses the
    previous prediction whenever the third-order finite difference of the input latent trajectory is
    bounded by `threshold`. The bound is estimated from the latents alone, so deciding to skip adds
    no model evaluation.
    """

    _is_stateful = True

    def __init__(self, state_manager: StateManager, config: AdaptiveDiffusionConfig):
        self.state_manager = state_manager
        self.config = config

    @torch.compiler.disable
    def new_forward(self, module: torch.nn.Module, *args, **kwargs):
        if self.state_manager._current_context is None:
            self.state_manager.set_context("inference")

        hidden_states = self._get_hidden_states(args, kwargs)
        state: AdaptiveDiffusionState = self.state_manager.get_state()

        if self._should_skip(hidden_states, state):
            logger.debug(f"AdaptiveDiffusion: reusing cached prediction at step {state.step_index}")
            output = state.cached_output
            state.consecutive_skips += 1
        else:
            output = self.fn_ref.original_forward(*args, **kwargs)
            state.cached_output = output
            state.consecutive_skips = 0

        state.latent_history.append(hidden_states)
        self._advance_step(state)
        return output

    def reset_state(self, module):
        self.state_manager.reset()
        return module

    def _get_hidden_states(self, args, kwargs):
        if "hidden_states" in kwargs:
            return kwargs["hidden_states"]
        return args[0]

    def _should_skip(self, hidden_states: torch.Tensor, state: AdaptiveDiffusionState) -> bool:
        # A third-order difference needs the current latent plus the three preceding ones. The first
        # few steps therefore always compute, which also warms the cache.
        if len(state.latent_history) < 3 or state.cached_output is None:
            return False
        if state.consecutive_skips >= self.config.max_skip_steps:
            return False

        x_prev = state.latent_history[-1]
        x_prev2 = state.latent_history[-2]
        x_prev3 = state.latent_history[-3]
        third_order = hidden_states - 3.0 * x_prev + 3.0 * x_prev2 - x_prev3
        bounded_diff = (third_order.abs().mean() / (hidden_states.abs().mean() + 1e-8)).item()
        return bounded_diff < self.config.threshold

    def _advance_step(self, state: AdaptiveDiffusionState):
        state.step_index += 1
        if state.step_index >= self.config.num_inference_steps:
            # End of a denoising loop: clear the rolling cache for the next generation.
            state.latent_history.clear()
            state.cached_output = None
            state.step_index = 0
            state.consecutive_skips = 0


def apply_adaptive_diffusion(module: torch.nn.Module, config: AdaptiveDiffusionConfig) -> None:
    """
    Applies [AdaptiveDiffusion](https://huggingface.co/papers/2410.09873) to a given module.

    AdaptiveDiffusion is a training-free step cache: it wraps the module's forward (the per-step
    noise prediction) and reuses the previous prediction whenever the third-order finite difference
    of the input latent trajectory is bounded by `config.threshold`. The bound is estimated from the
    latents alone, so the skip decision adds no model evaluation.

    Args:
        module (`torch.nn.Module`):
            The module to wrap. This should be the denoising model called inside the pipeline's
            denoising loop, such as a Diffusers transformer (`pipe.transformer`).
        config (`AdaptiveDiffusionConfig`):
            The configuration to use.

    Example:
        ```python
        >>> import torch
        >>> from diffusers import FluxPipeline
        >>> from diffusers.hooks.adaptive_diffusion import AdaptiveDiffusionConfig, apply_adaptive_diffusion

        >>> pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16)
        >>> pipe.to("cuda")

        >>> apply_adaptive_diffusion(pipe.transformer, AdaptiveDiffusionConfig(threshold=0.05))

        >>> image = pipe("a photo of an astronaut riding a horse", generator=torch.Generator().manual_seed(0)).images[0]
        ```
    """

    HookRegistry.check_if_exists_or_initialize(module)

    state_manager = StateManager(AdaptiveDiffusionState, (), {})
    registry = HookRegistry.check_if_exists_or_initialize(module)
    if registry.get_hook(_ADAPTIVE_DIFFUSION_HOOK) is not None:
        registry.remove_hook(_ADAPTIVE_DIFFUSION_HOOK)

    hook = AdaptiveDiffusionHook(state_manager, config)
    registry.register_hook(hook, _ADAPTIVE_DIFFUSION_HOOK)
