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

import re
from dataclasses import dataclass
from typing import Any, Callable

import torch

from ..utils import logging
from .hooks import HookRegistry, ModelHook


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


_LOW_RES_FEATURE_CACHE_BLOCK_HOOK = "low_res_feature_cache_block"

# Candidate block containers whose elements may be cached. Covers both UNet-style
# multi-resolution stacks (down/up blocks) and DiT transformer-block containers.
_LOW_RESOLUTION_BLOCK_IDENTIFIERS = (
    "down_blocks",
    "up_blocks",
    "transformer_blocks",
    "single_transformer_blocks",
    "blocks",
)


@dataclass
class LowResFeatureCacheConfig:
    r"""
    Configuration for the low-resolution feature cache, an inference-time acceleration adapted
    from [Clockwork Diffusion](https://huggingface.co/papers/2312.08128).

    Clockwork Diffusion observes that UNet layers operating on high-resolution feature maps are
    sensitive to small perturbations and must be recomputed at every denoising step, whereas
    layers operating on low-resolution feature maps govern the semantic layout and change slowly
    across steps — so their output can be reused from a preceding step and recomputed only
    periodically. This hook applies that idea as resolution-stratified reuse: the candidate blocks
    are stratified by the spatial resolution of their feature maps, and only the low-resolution
    stratum is cached and reused across denoising steps.

    Attributes:
        low_resolution_block_skip_range (`int`, defaults to `3`):
            Recompute the cached low-resolution blocks every `N` denoising steps; their output is
            reused as-is for the `N - 1` steps in between. Must be at least `1`; a value of `1`
            disables reuse (every block is recomputed every step).
        low_resolution_spatial_fraction (`float`, defaults to `0.5`):
            A candidate block is treated as low-resolution (and thus cacheable) only if the spatial
            area of its feature map is at most this fraction of the largest candidate block's spatial
            area. This parameter-free stratification is what separates the low-resolution stratum that
            Clockwork reuses from the high-resolution stratum it always recomputes.
        timestep_skip_range (`tuple[float, float]`, defaults to `(-1.0, inf)`):
            The denoising-timestep window within which reuse is allowed. Outside this window every
            block is recomputed, which protects the high-noise early steps where outputs change most.
        low_resolution_block_identifiers (`tuple[str, ...]`, defaults to the standard block containers):
            Regex patterns matched against module names to select the candidate blocks that may be
            cached. Matching uses `re.search`, so partial names and patterns are supported.
        current_timestep_callback (`Callable[[], float]`, defaults to `None`):
            Returns the current denoising timestep. Required only when `timestep_skip_range` is
            narrower than the full schedule; if left as `None` the timestep window is ignored and
            reuse is permitted at every step.
    """

    low_resolution_block_skip_range: int = 3
    low_resolution_spatial_fraction: float = 0.5
    timestep_skip_range: tuple[float, float] = (-1.0, float("inf"))
    low_resolution_block_identifiers: tuple[str, ...] = _LOW_RESOLUTION_BLOCK_IDENTIFIERS
    current_timestep_callback: Callable[[], float] = None

    def __repr__(self) -> str:
        return (
            f"LowResFeatureCacheConfig(\n"
            f"  low_resolution_block_skip_range={self.low_resolution_block_skip_range},\n"
            f"  low_resolution_spatial_fraction={self.low_resolution_spatial_fraction},\n"
            f"  timestep_skip_range={self.timestep_skip_range},\n"
            f"  low_resolution_block_identifiers={self.low_resolution_block_identifiers},\n"
            f")"
        )


class LowResFeatureCacheSharedState:
    r"""
    Holds the largest feature-map spatial area seen across candidate blocks. Shared by every block a
    single config is applied to so each block can decide whether it belongs to the low-resolution
    stratum. The maximum is only ever grown (blocks are only ever demoted from high-res to low-res as
    it grows), so the stratification converges to the correct split after the first denoiser pass.
    """

    def __init__(self) -> None:
        self.max_spatial_area: int | None = None


class LowResFeatureCacheBlockState:
    r"""
    Per-block state for the low-resolution feature cache. `spatial_area` is measured once and kept
    across generations (a model's block resolutions do not change between runs at a fixed resolution).
    """

    def __init__(self) -> None:
        self.iteration: int = 0
        self.spatial_area: int | None = None
        self.cache: Any = None

    def reset(self) -> None:
        self.iteration = 0
        self.cache = None


class LowResFeatureCacheBlockHook(ModelHook):
    _is_stateful = True

    def __init__(
        self,
        block_skip_range: int,
        spatial_fraction: float,
        timestep_skip_range: tuple[float, float],
        current_timestep_callback: Callable[[], float],
        shared_state: LowResFeatureCacheSharedState,
    ) -> None:
        super().__init__()

        self.block_skip_range = block_skip_range
        self.spatial_fraction = spatial_fraction
        self.timestep_skip_range = timestep_skip_range
        self.current_timestep_callback = current_timestep_callback
        self.shared_state = shared_state

    def initialize_hook(self, module):
        self.state = LowResFeatureCacheBlockState()
        return module

    @staticmethod
    def _get_hidden_states(args: tuple, kwargs: dict) -> torch.Tensor | None:
        if "hidden_states" in kwargs and torch.is_tensor(kwargs["hidden_states"]):
            return kwargs["hidden_states"]
        for arg in args:
            if torch.is_tensor(arg):
                return arg
        return None

    @staticmethod
    def _spatial_area(hidden_states: torch.Tensor) -> int:
        # Resolution proxy: H*W for conv feature maps (B, C, H, W), sequence length for token
        # sequences (B, L, D). Smaller means lower resolution.
        if hidden_states.ndim == 4:
            return int(hidden_states.shape[2] * hidden_states.shape[3])
        return int(hidden_states.shape[1])

    def _is_low_resolution(self) -> bool:
        if self.shared_state.max_spatial_area is None or self.state.spatial_area is None:
            return False
        return self.state.spatial_area <= self.spatial_fraction * self.shared_state.max_spatial_area

    def _is_within_timestep_range(self) -> bool:
        if self.current_timestep_callback is None:
            return True
        timestep = self.current_timestep_callback()
        return self.timestep_skip_range[0] < timestep < self.timestep_skip_range[1]

    def new_forward(self, module: torch.nn.Module, *args, **kwargs) -> Any:
        hidden_states = self._get_hidden_states(args, kwargs)
        if hidden_states is not None:
            # Measure once and grow the shared maximum. Because the maximum only ever increases,
            # a block can only be demoted from high-res to low-res — never the reverse — so the
            # stratification is correct once the first denoiser pass has completed.
            if self.state.spatial_area is None:
                self.state.spatial_area = self._spatial_area(hidden_states)
            if self.shared_state.max_spatial_area is None or (
                self.state.spatial_area > self.shared_state.max_spatial_area
            ):
                self.shared_state.max_spatial_area = self.state.spatial_area

        is_low_resolution = self._is_low_resolution()
        # The first pass (iteration 0) always computes so it can seed the cache; reuse only applies
        # to the low-resolution stratum, on non-recompute steps, within the timestep window.
        should_reuse = (
            is_low_resolution
            and self.state.cache is not None
            and self.state.iteration > 0
            and self.state.iteration % self.block_skip_range != 0
            and self._is_within_timestep_range()
        )

        if should_reuse:
            logger.debug("LowResFeatureCache - reusing cached low-resolution block output")
            output = self.state.cache
        else:
            output = self.fn_ref.original_forward(*args, **kwargs)
            # Cache only the stratum we will reuse; high-resolution blocks stay uncached so their
            # (large) outputs are never copied. The cache is a single slot, overwritten on recompute.
            self.state.cache = output if is_low_resolution else None

        self.state.iteration += 1
        return output

    def reset_state(self, module: torch.nn.Module) -> torch.nn.Module:
        self.state.reset()
        return module


def apply_low_res_feature_cache(module: torch.nn.Module, config: LowResFeatureCacheConfig) -> None:
    r"""
    Applies the low-resolution feature cache (adapted from
    [Clockwork Diffusion](https://huggingface.co/papers/2312.08128)) to a given module.

    The candidate blocks are the elements of any `nn.ModuleList` child whose name matches
    `config.low_resolution_block_identifiers`. Each candidate is stratified by the spatial resolution
    of its feature map, and only the low-resolution stratum is cached and reused across denoising
    steps while the high-resolution stratum is recomputed every step.

    Args:
        module (`torch.nn.Module`):
            The module to apply the cache to. Typically the denoiser of a pipeline, such as a
            `UNet2DConditionModel` or a Diffusers transformer, but external implementations may also
            work as long as their reusable blocks live in a named `nn.ModuleList`.
        config (`LowResFeatureCacheConfig`):
            The configuration to use.

    Example:
    ```python
    >>> import torch
    >>> from diffusers import StableDiffusionPipeline, LowResFeatureCacheConfig, apply_low_res_feature_cache

    >>> pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16)
    >>> pipe.to("cuda")

    >>> apply_low_res_feature_cache(pipe.unet, LowResFeatureCacheConfig(low_resolution_block_skip_range=3))
    ```
    """

    logger.warning(
        "The low-resolution feature cache is a purely experimental feature and may not work as "
        "expected. Not all models benefit from it. The API is subject to change in future releases, "
        "with no guarantee of backward compatibility. Please report any issues at "
        "https://github.com/huggingface/diffusers/issues."
    )

    if config.low_resolution_block_skip_range < 1:
        raise ValueError(
            f"`low_resolution_block_skip_range` must be at least 1, but got {config.low_resolution_block_skip_range}."
        )
    if not 0 < config.low_resolution_spatial_fraction <= 1.0:
        raise ValueError(
            "`low_resolution_spatial_fraction` must be in the interval (0, 1], but got "
            f"{config.low_resolution_spatial_fraction}."
        )

    shared_state = LowResFeatureCacheSharedState()
    num_candidate_blocks = 0

    for name, submodule in module.named_children():
        if not isinstance(submodule, torch.nn.ModuleList):
            continue
        if not any(re.search(identifier, name) is not None for identifier in config.low_resolution_block_identifiers):
            continue
        for index, block in enumerate(submodule):
            _apply_low_res_feature_cache_on_block(f"{name}.{index}", block, config, shared_state)
            num_candidate_blocks += 1

    if num_candidate_blocks == 0:
        logger.warning(
            "No candidate blocks were found to apply the low-resolution feature cache to. "
            "This usually means the module does not expose its reusable blocks as a named "
            "`nn.ModuleList` matching `low_resolution_block_identifiers`."
        )


def _apply_low_res_feature_cache_on_block(
    name: str, block: torch.nn.Module, config: LowResFeatureCacheConfig, shared_state: LowResFeatureCacheSharedState
) -> None:
    logger.debug(f"Applying LowResFeatureCacheBlockHook to '{name}'")
    hook = LowResFeatureCacheBlockHook(
        config.low_resolution_block_skip_range,
        config.low_resolution_spatial_fraction,
        config.timestep_skip_range,
        config.current_timestep_callback,
        shared_state,
    )
    registry = HookRegistry.check_if_exists_or_initialize(block)
    registry.register_hook(hook, _LOW_RES_FEATURE_CACHE_BLOCK_HOOK)
