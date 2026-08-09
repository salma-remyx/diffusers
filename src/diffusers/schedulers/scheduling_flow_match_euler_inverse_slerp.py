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

import math
from dataclasses import dataclass
from typing import Literal, Union

import numpy as np
import torch

from ..configuration_utils import ConfigMixin, register_to_config
from ..utils import BaseOutput
from .scheduling_utils import SchedulerMixin


@dataclass
class FlowMatchEulerInverseSlerpSchedulerOutput(BaseOutput):
    """
    Output class for the scheduler's `step` function output.

    Args:
        prev_sample (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)` for images):
            Computed sample at the next (noisier) timestep of the inversion chain. `prev_sample` should be used as the
            next model input.
    """

    prev_sample: torch.FloatTensor


class FlowMatchEulerInverseSlerpScheduler(SchedulerMixin, ConfigMixin):
    """
    Inverse flow-matching Euler scheduler with SlerpFlow spherical velocity correction.

    This is the rectified-flow analog of [`DDIMInverseScheduler`]: it walks the noise schedule from data (`sigma=0`)
    toward noise (`sigma=1`), inverting a rectified-flow model (e.g. FLUX) so an image latent can be mapped back to the
    noise that a forward pass would reconstruct it from.

    The update rule applies the SlerpFlow geometric correction
    (https://arxiv.org/abs/2607.21326v1): discretization error bends the rectified-flow trajectory off the data
    manifold, and that bend is treated as a necessary "centripetal force". The velocity direction is therefore
    rectified on a hypersphere via Spherical Linear Interpolation (Slerp) between the current and the previously cached
    velocity, then the step is taken with a first-order Euler update. Caching the previous velocity means every step
    still costs a single model evaluation, preserving the efficiency of a plain Euler solver while improving inversion
    fidelity.

    Adapted from SlerpFlow (Mode 2 port): the paper evaluates the second velocity at the Euler-guessed sample
    (a predictor/corrector pair) and reuses it for the next step. This scheduler instead feeds the pipeline's
    per-step model output in directly and reuses it as the next step's cached velocity -- the two coincide in the
    first-order regime the method targets, so the core mechanism (Slerp direction correction + velocity reuse at one
    evaluation per step) is preserved while fitting the standard single-call-per-step scheduler contract.

    Args:
        num_train_timesteps (`int`, defaults to 1000):
            The number of diffusion steps to train the model.
        shift (`float`, defaults to 1.0):
            The shift value for the timestep schedule.
        use_dynamic_shifting (`bool`, defaults to False):
            Whether to apply timestep shifting on-the-fly based on the image resolution.
        base_shift (`float`, defaults to 0.5):
            Value to stabilize image generation when dynamic shifting is enabled.
        max_shift (`float`, defaults to 1.15):
            Value change allowed to latent vectors when dynamic shifting is enabled.
        base_image_seq_len (`int`, defaults to 256):
            The base image sequence length for dynamic shifting.
        max_image_seq_len (`int`, defaults to 4096):
            The maximum image sequence length for dynamic shifting.
        time_shift_type (`str`, defaults to `"exponential"`):
            The type of dynamic resolution-dependent timestep shifting to apply. Either `"exponential"` or `"linear"`.
        slerp_t (`float`, defaults to 0.5):
            SlerpFlow interpolation fraction between the cached velocity direction (`0.0`) and the current velocity
            direction (`1.0`). `0.5` applies an equal angular blend, matching the paper default.
    """

    _compatibles = []
    order = 1

    @register_to_config
    def __init__(
        self,
        num_train_timesteps: int = 1000,
        shift: float = 1.0,
        use_dynamic_shifting: bool = False,
        base_shift: float = 0.5,
        max_shift: float = 1.15,
        base_image_seq_len: int = 256,
        max_image_seq_len: int = 4096,
        time_shift_type: Literal["exponential", "linear"] = "exponential",
        slerp_t: float = 0.5,
    ):
        if time_shift_type not in {"exponential", "linear"}:
            raise ValueError("`time_shift_type` must either be 'exponential' or 'linear'.")

        timesteps = np.linspace(1, num_train_timesteps, num_train_timesteps, dtype=np.float32)[::-1].copy()
        timesteps = torch.from_numpy(timesteps).to(dtype=torch.float32)

        sigmas = timesteps / num_train_timesteps
        if not use_dynamic_shifting:
            # when use_dynamic_shifting is True, shifting is applied on the fly based on the image resolution
            sigmas = shift * sigmas / (1 + (shift - 1) * sigmas)

        self.timesteps = sigmas * num_train_timesteps
        self.sigmas = sigmas.to("cpu")  # to avoid too much CPU/GPU communication
        self.sigma_min = self.sigmas[-1].item()
        self.sigma_max = self.sigmas[0].item()

        self.init_noise_sigma = 1.0

        self._shift = shift

        self._step_index = None
        self._begin_index = None
        # Velocity cached from the previous step; reused as the "current" velocity for the Slerp correction so each
        # step needs only one model evaluation.
        self._cached_velocity = None

    @property
    def shift(self):
        """
        The value used for shifting.
        """
        return self._shift

    @property
    def step_index(self):
        """
        The index counter for current timestep. It will increase 1 after each scheduler step.
        """
        return self._step_index

    @property
    def begin_index(self):
        """
        The index for the first timestep. It should be set from pipeline with `set_begin_index` method.
        """
        return self._begin_index

    def set_begin_index(self, begin_index: int = 0):
        """
        Sets the begin index for the scheduler. This function should be run from pipeline before the inference.

        Args:
            begin_index (`int`, defaults to `0`):
                The begin index for the scheduler.
        """
        self._begin_index = begin_index

    def _sigma_to_t(self, sigma) -> float:
        return sigma * self.config.num_train_timesteps

    def time_shift(self, mu: float, sigma: float, t: torch.Tensor) -> torch.Tensor:
        """
        Apply time shifting to the sigmas.

        Args:
            mu (`float`):
                The mu parameter for the time shift.
            sigma (`float`):
                The sigma parameter for the time shift.
            t (`torch.Tensor`):
                The input timesteps.

        Returns:
            `torch.Tensor`:
                The time-shifted timesteps.
        """
        if self.config.time_shift_type == "exponential":
            return self._time_shift_exponential(mu, sigma, t)
        return self._time_shift_linear(mu, sigma, t)

    def set_timesteps(
        self,
        num_inference_steps: int,
        device: str | torch.device | None = None,
        mu: float | None = None,
    ):
        """
        Sets the discrete timesteps used for the inversion chain (to be run before inference). The schedule runs from
        data (`sigma_min`) to noise (`sigma_max`), the reverse of the forward `FlowMatchEulerDiscreteScheduler`.

        Args:
            num_inference_steps (`int`):
                The number of diffusion steps used when inverting a sample with a pre-trained model.
            device (`str` or `torch.device`, *optional*):
                The device to which the timesteps should be moved. If `None`, the timesteps are not moved.
            mu (`float`, *optional*):
                Determines the amount of shifting applied to sigmas when performing resolution-dependent timestep
                shifting. Must be passed when `use_dynamic_shifting=True`.
        """
        if self.config.use_dynamic_shifting and mu is None:
            raise ValueError("`mu` must be passed when `use_dynamic_shifting` is set to be `True`")

        self.num_inference_steps = num_inference_steps

        # Same decreasing sigma grid (noise -> data) as the forward FlowMatchEulerDiscreteScheduler.
        timesteps = np.linspace(
            self._sigma_to_t(self.sigma_max), self._sigma_to_t(self.sigma_min), num_inference_steps
        )
        sigmas = timesteps / self.config.num_train_timesteps
        if self.config.use_dynamic_shifting:
            sigmas = self.time_shift(mu, 1.0, sigmas)
        else:
            sigmas = self.shift * sigmas / (1 + (self.shift - 1) * sigmas)

        # Invert the grid (data -> noise); these are the model evaluation points, the exact reverse of the forward
        # FlowMatchEulerDiscreteScheduler. The pure-noise terminal (sigma=1) is appended to `sigmas` only, mirroring the
        # forward scheduler's sigma=0 terminal, so the final step lands on the noise endpoint.
        sigmas = torch.from_numpy(np.flip(np.array(sigmas, dtype=np.float32)).copy()).to(
            device=device, dtype=torch.float32
        )
        timesteps = sigmas * self.config.num_train_timesteps
        sigmas = torch.cat([sigmas, torch.ones(1, device=sigmas.device)])

        self.timesteps = timesteps
        self.sigmas = sigmas
        self._step_index = None
        self._begin_index = None
        self._cached_velocity = None

    def index_for_timestep(
        self,
        timestep: Union[float, torch.FloatTensor],
        schedule_timesteps: torch.FloatTensor | None = None,
    ) -> int:
        if schedule_timesteps is None:
            schedule_timesteps = self.timesteps

        indices = (schedule_timesteps == timestep).nonzero()
        pos = 1 if len(indices) > 1 else 0
        return indices[pos].item()

    def _init_step_index(self, timestep: Union[float, torch.FloatTensor]) -> None:
        if self.begin_index is None:
            if isinstance(timestep, torch.Tensor):
                timestep = timestep.to(self.timesteps.device)
            self._step_index = self.index_for_timestep(timestep)
        else:
            self._step_index = self._begin_index

    def scale_model_input(self, sample: torch.FloatTensor, timestep: int | None = None) -> torch.FloatTensor:
        """
        Ensures interchangeability with schedulers that need to scale the denoising model input depending on the
        current timestep.

        Args:
            sample (`torch.FloatTensor`):
                The input sample.
            timestep (`int`, *optional*):
                The current timestep in the diffusion chain.

        Returns:
            `torch.FloatTensor`:
                The same input sample (flow-matching needs no input scaling).
        """
        return sample

    def _slerp_velocity(self, v_curr: torch.Tensor, v_next: torch.Tensor, slerp_t: float) -> torch.Tensor:
        """
        SlerpFlow spatial-local spherical correction of the velocity direction.

        Interpolates the velocity direction on the per-position hypersphere between the cached velocity `v_curr` and
        the current velocity `v_next`, keeping the current velocity's magnitude. Reduces to a linear blend when the two
        directions are (nearly) parallel.

        Args:
            v_curr (`torch.Tensor`):
                The cached velocity from the previous step.
            v_next (`torch.Tensor`):
                The velocity predicted by the model at the current step.
            slerp_t (`float`):
                Interpolation fraction between `v_curr` (0.0) and `v_next` (1.0).

        Returns:
            `torch.Tensor`:
                The direction-corrected velocity to integrate with the Euler step.
        """
        # Reduce over channels for image latents [B, C, H, W]; over the last axis otherwise (e.g. sequence form).
        dim_to_reduce = 1 if v_curr.ndim == 4 else -1

        norm_curr = torch.norm(v_curr, p=2, dim=dim_to_reduce, keepdim=True)
        norm_next = torch.norm(v_next, p=2, dim=dim_to_reduce, keepdim=True)

        dir_curr = v_curr / (norm_curr + 1e-6)
        dir_next = v_next / (norm_next + 1e-6)

        dot = (dir_curr * dir_next).sum(dim=dim_to_reduce, keepdim=True)
        dot = torch.clamp(dot, -1 + 1e-5, 1 - 1e-5)
        theta = torch.acos(dot)
        sin_theta = torch.sin(theta)

        # Near-parallel directions collapse to a linear blend (sin(theta) -> 0 makes the slerp weights ill-defined).
        is_parallel = sin_theta < 1e-4
        w_next = torch.where(
            is_parallel,
            torch.full_like(sin_theta, slerp_t),
            torch.sin((1.0 - slerp_t) * theta) / (sin_theta + 1e-6),
        )
        w_curr = torch.where(
            is_parallel,
            torch.full_like(sin_theta, 1.0 - slerp_t),
            torch.sin(slerp_t * theta) / (sin_theta + 1e-6),
        )

        dir_final = w_next * dir_next + w_curr * dir_curr
        return dir_final * norm_next

    def step(
        self,
        model_output: torch.FloatTensor,
        timestep: float | torch.FloatTensor,
        sample: torch.FloatTensor,
        return_dict: bool = True,
    ) -> FlowMatchEulerInverseSlerpSchedulerOutput | tuple:
        """
        Predict the noisier sample at the next timestep of the inversion chain.

        The SlerpFlow correction combines the direction of the cached velocity (reused from the previous step) with the
        direction of the current model output, then integrates the corrected velocity with a first-order Euler step.
        On the first step there is no cached velocity yet, so a plain Euler step is taken and the velocity is cached for
        the next step.

        Args:
            model_output (`torch.FloatTensor`):
                The velocity predicted by the learned rectified-flow model.
            timestep (`float` or `torch.FloatTensor`):
                The current discrete timestep in the inversion chain.
            sample (`torch.FloatTensor`):
                A current instance of a sample being inverted.
            return_dict (`bool`, defaults to `True`):
                Whether or not return a
                [`~schedulers.scheduling_flow_match_euler_inverse_slerp.FlowMatchEulerInverseSlerpSchedulerOutput`] or
                tuple.

        Returns:
            [`~schedulers.scheduling_flow_match_euler_inverse_slerp.FlowMatchEulerInverseSlerpSchedulerOutput`] or
            `tuple`: If `return_dict` is `True`, the output dataclass is returned, otherwise a tuple whose first element
            is the noisier sample tensor.
        """
        if (
            isinstance(timestep, int)
            or isinstance(timestep, torch.IntTensor)
            or isinstance(timestep, torch.LongTensor)
        ):
            raise ValueError(
                (
                    "Passing integer indices (e.g. from `enumerate(timesteps)`) as timesteps to"
                    " `FlowMatchEulerInverseSlerpScheduler.step()` is not supported. Make sure to pass"
                    " one of the `scheduler.timesteps` as a timestep."
                ),
            )

        if self.step_index is None:
            self._init_step_index(timestep)

        # Upcast to avoid precision issues when computing prev_sample
        sample = sample.to(torch.float32)
        model_output = model_output.to(torch.float32)

        sigma = self.sigmas[self.step_index]
        sigma_next = self.sigmas[self.step_index + 1]
        # Inversion walks toward larger sigma, so dt > 0 (adding structure/noise rather than removing it).
        dt = sigma_next - sigma

        v_next = model_output
        v_curr = self._cached_velocity
        if v_curr is None:
            # First step: no cached velocity to correct against, fall back to plain Euler.
            v_final = v_next
        else:
            v_final = self._slerp_velocity(v_curr, v_next, self.config.slerp_t)

        prev_sample = sample + dt * v_final

        # Reuse this step's velocity as the next step's "current" velocity.
        self._cached_velocity = v_next

        # Upon completion increase step index by one and cast back to model compatible dtype.
        self._step_index += 1
        prev_sample = prev_sample.to(model_output.dtype)

        if not return_dict:
            return (prev_sample,)

        return FlowMatchEulerInverseSlerpSchedulerOutput(prev_sample=prev_sample)

    def _time_shift_exponential(self, mu: float, sigma: float, t: torch.Tensor) -> torch.Tensor:
        return math.exp(mu) / (math.exp(mu) + (1 / t - 1) ** sigma)

    def _time_shift_linear(self, mu: float, sigma: float, t: torch.Tensor) -> torch.Tensor:
        return mu / (mu + (1 / t - 1) ** sigma)

    def __len__(self) -> int:
        return self.config.num_train_timesteps
