# Copyright 2026 HuggingFace Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
# the License. You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
# an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.

import gc

import pytest
import torch

from diffusers.hooks import HookRegistry
from diffusers.hooks.losa_sparse_attention import LoSAConfig, apply_losa_sparse_attention
from diffusers.models.attention_processor import Attention


class TestLoSASparseAttention:
    query_dim = 32
    heads = 4
    dim_head = 8
    sequence_length = 16
    block_size = 4

    def teardown_method(self):
        gc.collect()

    def _build_self_attention(self):
        # Plain self-attention matching the standard video-DiT configuration: MHA, qk scaling, no group/spatial norm.
        torch.manual_seed(0)
        return Attention(query_dim=self.query_dim, heads=self.heads, dim_head=self.dim_head).eval()

    def _diffuse_hidden_states(self, batch_size=2):
        # Random tokens: attention is diffuse, so a high retention threshold keeps every block (no sparsity).
        torch.manual_seed(1)
        return torch.randn(batch_size, self.sequence_length, self.query_dim)

    def _block_local_hidden_states(self, batch_size=2):
        # Tokens within a block share a (well-separated) centroid. This mimics the locality of real video
        # self-attention, where each query block concentrates its mass on a few key blocks. Under this input a 0.99
        # retention threshold actually drops blocks while staying near-lossless.
        num_blocks = (self.sequence_length + self.block_size - 1) // self.block_size
        generator = torch.Generator().manual_seed(42)
        centroids = torch.randn(num_blocks, self.query_dim, generator=generator) * 5.0
        block_of_token = [min(t // self.block_size, num_blocks - 1) for t in range(self.sequence_length)]
        base = torch.stack([centroids[block_index] for block_index in block_of_token])
        noise = torch.randn(batch_size, self.sequence_length, self.query_dim, generator=generator) * 0.05
        return base.unsqueeze(0).expand(batch_size, -1, -1) + noise

    @staticmethod
    def _relative_peak_error(a: torch.Tensor, b: torch.Tensor) -> float:
        # Peak absolute deviation normalized by the peak magnitude of the reference, so the bound is scale-free.
        return (a - b).abs().max().item() / b.abs().max().item()

    def _selected_positions(self, hook, batch_heads):
        num_query_blocks = (self.sequence_length + self.block_size - 1) // self.block_size
        kept = sum(int(validity.sum().item()) for _, validity in hook.state.selected)
        return kept, batch_heads * num_query_blocks * self.sequence_length

    def test_profile_step_matches_dense_attention(self):
        # The profile step is dense by construction, so it must reproduce the unhooked processor output exactly.
        attn = self._build_self_attention()
        hidden_states = self._diffuse_hidden_states()

        with torch.no_grad():
            dense_output = attn(hidden_states)

        apply_losa_sparse_attention(attn, LoSAConfig(retained_mass_threshold=0.99, block_size=self.block_size))
        registry = HookRegistry.check_if_exists_or_initialize(attn)
        assert registry.get_hook("losa_sparse_attention") is not None

        with torch.no_grad():
            profile_output = attn(hidden_states)

        assert torch.allclose(profile_output, dense_output, atol=1e-5)

    def test_sparse_step_is_near_lossless_with_sparsity(self):
        # Under concentrated attention, retaining 99% of the mass drops blocks AND stays near-lossless.
        attn = self._build_self_attention()
        reference = self._build_self_attention()
        reference.load_state_dict(attn.state_dict())
        hidden_states = self._block_local_hidden_states()

        apply_losa_sparse_attention(attn, LoSAConfig(retained_mass_threshold=0.99, block_size=self.block_size))
        hook = HookRegistry.check_if_exists_or_initialize(attn).get_hook("losa_sparse_attention")

        with torch.no_grad():
            attn(hidden_states)  # iteration 0: profile
            sparse_output = attn(hidden_states)  # iteration 1: sparse reconstruction
            dense_output = reference(hidden_states)

        # Selection is frozen after the profile pass and reused on later steps (profile-once-freeze).
        assert hook.state.profiled is True
        assert hook.state.iteration == 2
        assert hook.state.selected is not None

        # Sparsity is achieved (some key blocks are dropped) yet the reconstruction stays near-lossless.
        kept, full = self._selected_positions(hook, batch_heads=2 * self.heads)
        assert 0 < kept < full
        assert self._relative_peak_error(sparse_output, dense_output) < 0.1

    def test_aggressive_threshold_trades_fidelity_for_sparsity(self):
        # A looser threshold drops more blocks and is no longer near-lossless: the speed/quality trade-off.
        attn = self._build_self_attention()
        reference = self._build_self_attention()
        reference.load_state_dict(attn.state_dict())
        hidden_states = self._block_local_hidden_states()

        apply_losa_sparse_attention(attn, LoSAConfig(retained_mass_threshold=0.5, block_size=self.block_size))
        hook = HookRegistry.check_if_exists_or_initialize(attn).get_hook("losa_sparse_attention")

        with torch.no_grad():
            attn(hidden_states)  # profile
            sparse_output = attn(hidden_states)  # sparse
            dense_output = reference(hidden_states)

        kept, full = self._selected_positions(hook, batch_heads=2 * self.heads)
        assert 0 < kept < full
        assert self._relative_peak_error(sparse_output, dense_output) > 0.15

    def test_reset_clears_frozen_selection(self):
        # Between generations the pipeline resets stateful hooks; LoSA must re-profile on the next pass.
        attn = self._build_self_attention()
        hidden_states = self._diffuse_hidden_states()

        apply_losa_sparse_attention(attn, LoSAConfig(retained_mass_threshold=0.99, block_size=self.block_size))
        registry = HookRegistry.check_if_exists_or_initialize(attn)
        hook = registry.get_hook("losa_sparse_attention")

        with torch.no_grad():
            attn(hidden_states)
        assert hook.state.profiled is True

        registry.reset_stateful_hooks()
        assert hook.state.profiled is False
        assert hook.state.selected is None

    def test_cross_attention_is_not_hooked(self):
        # LoSA targets the long self-attention sequence; cross-attention layers are skipped at apply time.
        cross_attn = Attention(query_dim=self.query_dim, cross_attention_dim=64, heads=self.heads, dim_head=self.dim_head)
        apply_losa_sparse_attention(cross_attn, LoSAConfig(retained_mass_threshold=0.99, block_size=self.block_size))

        registry = HookRegistry.check_if_exists_or_initialize(cross_attn)
        assert registry.get_hook("losa_sparse_attention") is None

    def test_invalid_config_raises(self):
        with pytest.raises(ValueError):
            LoSAConfig(retained_mass_threshold=0.0)
        with pytest.raises(ValueError):
            LoSAConfig(retained_mass_threshold=1.5)
        with pytest.raises(ValueError):
            LoSAConfig(block_size=0)
