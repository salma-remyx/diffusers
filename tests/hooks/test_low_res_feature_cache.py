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

import pytest
import torch

# Imports the public API surface (exercising the __init__.py wiring), not the new module directly.
from diffusers import LowResFeatureCacheConfig, apply_low_res_feature_cache
from diffusers.models import ModelMixin


class DoublerBlock(torch.nn.Module):
    """A toy block: output is twice the input. Lets us tell compute (2 * input) apart from reuse."""

    def forward(self, hidden_states, **kwargs):
        return hidden_states * 2.0


class MultiResolutionUNet(ModelMixin):
    """Stand-in for a UNet: the first block sees a high-res feature map, the second a downsampled
    low-res feature map. With spatial_fraction=0.5 only the low-res block qualifies for caching."""

    def __init__(self):
        super().__init__()
        self.down_blocks = torch.nn.ModuleList([DoublerBlock(), DoublerBlock()])

    def forward(self, hidden_states):
        high = self.down_blocks[0](hidden_states)
        low_input = hidden_states[:, :, ::2, ::2]
        low = self.down_blocks[1](low_input)
        return high, low


def _reset_stateful_hooks(model):
    for module in model.modules():
        if hasattr(module, "_diffusers_hook"):
            module._diffusers_hook.reset_stateful_hooks(recurse=False)


def test_low_res_block_is_reused_high_res_block_is_recomputed():
    """Core Clockwork behaviour: across denoising steps the low-resolution block reuses its cached
    output while the high-resolution block is recomputed every step."""
    model = MultiResolutionUNet()
    apply_low_res_feature_cache(model, LowResFeatureCacheConfig(low_resolution_block_skip_range=3))

    # Step 0: both blocks compute and seed the cache.
    high0 = torch.ones(1, 1, 8, 8)
    high_out_0, low_out_0 = model(high0)
    # high: 8x8=64 (max area); low input: 4x4=16 <= 0.5*64 -> low-res stratum, cached.
    assert torch.allclose(high_out_0, high0 * 2.0)
    assert torch.allclose(low_out_0, high0[:, :, ::2, ::2] * 2.0)

    # Step 1: different input. The low block must reuse its step-0 cache; the high block must recompute.
    high1 = torch.full((1, 1, 8, 8), 5.0)
    high_out_1, low_out_1 = model(high1)
    assert torch.allclose(high_out_1, high1 * 2.0), "high-res block should be recomputed every step"
    assert torch.allclose(low_out_1, low_out_0), "low-res block should reuse the cached step-0 output"

    # Every 3rd step is a recompute step (iteration % 3 == 0): the low block refreshes its cache.
    # forward 2 (iter 1) reused; forward 3 (iter 2) reused; forward 4 (iter 3) recomputes.
    high2 = torch.full((1, 1, 8, 8), 9.0)
    model(high2)  # forward 3: still reusing the step-0 cache
    _, low_out_recompute = model(high2)  # forward 4: recompute against high2
    assert torch.allclose(low_out_recompute, high2[:, :, ::2, ::2] * 2.0), (
        "low-res block should recompute on the periodic refresh step"
    )


class SingleResolutionTransformer(ModelMixin):
    """Stand-in for a DiT: both blocks consume the same token sequence at one resolution, so there is
    no low-resolution stratum and the hook must not cache anything."""

    def __init__(self):
        super().__init__()
        self.transformer_blocks = torch.nn.ModuleList([DoublerBlock(), DoublerBlock()])

    def forward(self, hidden_states):
        for block in self.transformer_blocks:
            hidden_states = block(hidden_states)
        return hidden_states


def test_no_reuse_when_no_resolution_variation():
    """When every candidate block runs at the same resolution there is no low-resolution stratum, so
    nothing is cached and every block recomputes every step."""
    model = SingleResolutionTransformer()
    apply_low_res_feature_cache(model, LowResFeatureCacheConfig(low_resolution_block_skip_range=3))

    seq0 = torch.ones(1, 4, 8)
    out_0 = model(seq0)
    assert torch.allclose(out_0, seq0 * 4.0), "step 0 should compute normally"
    seq1 = torch.full((1, 4, 8), 5.0)
    out_1 = model(seq1)
    assert torch.allclose(out_1, seq1 * 4.0), "single-resolution blocks should always recompute"


def test_reset_clears_cache():
    """After resetting stateful hooks the cache is dropped and the next forward recomputes."""
    model = MultiResolutionUNet()
    apply_low_res_feature_cache(model, LowResFeatureCacheConfig(low_resolution_block_skip_range=3))

    high0 = torch.ones(1, 1, 8, 8)
    _, low_out_0 = model(high0)

    _reset_stateful_hooks(model)

    # After reset, the low block recomputes against the new input rather than returning the stale cache.
    high1 = torch.full((1, 1, 8, 8), 7.0)
    _, low_out_1 = model(high1)
    assert torch.allclose(low_out_1, high1[:, :, ::2, ::2] * 2.0)


def test_fraction_one_caches_all_blocks():
    """With spatial_fraction=1.0 every candidate block qualifies as low-resolution, so the high-res
    block is cached and reused too."""
    model = MultiResolutionUNet()
    apply_low_res_feature_cache(
        model, LowResFeatureCacheConfig(low_resolution_block_skip_range=3, low_resolution_spatial_fraction=1.0)
    )

    high0 = torch.ones(1, 1, 8, 8)
    high_out_0, _ = model(high0)
    high1 = torch.full((1, 1, 8, 8), 5.0)
    high_out_1, _ = model(high1)
    assert torch.allclose(high_out_1, high_out_0), "with fraction=1.0 the high-res block is also reused"


def test_invalid_config_raises():
    with pytest.raises(ValueError):
        apply_low_res_feature_cache(MultiResolutionUNet(), LowResFeatureCacheConfig(low_resolution_block_skip_range=0))
    with pytest.raises(ValueError):
        apply_low_res_feature_cache(
            MultiResolutionUNet(), LowResFeatureCacheConfig(low_resolution_spatial_fraction=0.0)
        )
