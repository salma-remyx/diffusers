# Copyright 2026 HuggingFace Inc.
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

import torch

from diffusers.hooks import HookRegistry
from diffusers.hooks.adaptive_diffusion import AdaptiveDiffusionConfig, apply_adaptive_diffusion


class CountingTransformer(torch.nn.Module):
    """Minimal denoising model: doubles its input and counts how often the forward really runs."""

    def __init__(self):
        super().__init__()
        self.calls = 0

    def forward(self, hidden_states, encoder_hidden_states=None):
        self.calls += 1
        return hidden_states * 2.0


def _set_context(model, context_name):
    for module in model.modules():
        if hasattr(module, "_diffusers_hook"):
            module._diffusers_hook._set_context(context_name)


def test_adaptive_diffusion_registers_with_existing_registry():
    model = CountingTransformer()
    apply_adaptive_diffusion(model, AdaptiveDiffusionConfig(threshold=1.0, num_inference_steps=8))

    # The hook is wired into the existing HookRegistry, which is what actually wraps forward.
    hook = HookRegistry.check_if_exists_or_initialize(model).get_hook("adaptive_diffusion_hook")
    assert hook is not None
    assert model(torch.tensor([[[1.0]]])).item() == 2.0


def test_adaptive_diffusion_skips_predictable_steps():
    model = CountingTransformer()
    # threshold=1.0 lets any small bounded difference skip; x_t = t**2 has an exactly-zero
    # third-order finite difference, so every step after the warmup should be reused.
    config = AdaptiveDiffusionConfig(threshold=1.0, max_skip_steps=5, num_inference_steps=8)
    apply_adaptive_diffusion(model, config)
    _set_context(model, "ctx")

    for t in range(6):
        out = model(torch.tensor([[[float(t * t)]]]))

    # Steps 0, 1, 2 compute (a third-order difference needs four latents); 3, 4, 5 are reused.
    assert model.calls == 3
    # The reused output is the last computed prediction (step 2: input 4 -> output 8.0).
    assert torch.allclose(out, torch.tensor([[[8.0]]]))


def test_adaptive_diffusion_computes_when_trajectory_is_unbounded():
    model = CountingTransformer()
    # A cubic trajectory has a non-zero third-order difference, which exceeds this tight threshold.
    config = AdaptiveDiffusionConfig(threshold=1e-6, max_skip_steps=5, num_inference_steps=8)
    apply_adaptive_diffusion(model, config)
    _set_context(model, "ctx")

    for t in range(6):
        model(torch.tensor([[[float(t**3)]]]))

    assert model.calls == 6


def test_adaptive_diffusion_max_skip_forces_recompute():
    model = CountingTransformer()
    # Always willing to skip, but never more than two steps in a row.
    config = AdaptiveDiffusionConfig(threshold=1.0, max_skip_steps=2, num_inference_steps=8)
    apply_adaptive_diffusion(model, config)
    _set_context(model, "ctx")

    for t in range(6):
        model(torch.tensor([[[float(t * t)]]]))

    # Warmup computes steps 0,1,2; then skip, skip, recompute -> 4 forward calls.
    assert model.calls == 4


def test_adaptive_diffusion_resets_between_generations():
    model = CountingTransformer()
    config = AdaptiveDiffusionConfig(threshold=1.0, max_skip_steps=5, num_inference_steps=4)
    apply_adaptive_diffusion(model, config)
    _set_context(model, "ctx")

    for t in range(4):  # one full generation: steps 0,1,2 compute, step 3 reuses
        model(torch.tensor([[[float(t * t)]]]))
    assert model.calls == 3

    for t in range(4):  # second generation: cache was reset, so steps 0,1,2 compute again
        model(torch.tensor([[[float(t * t)]]]))
    assert model.calls == 6
