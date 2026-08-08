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

from __future__ import annotations

from dataclasses import dataclass

import torch

from ..configuration_utils import ConfigMixin, register_to_config
from ..utils import BaseOutput
from .scheduling_utils import KarrasDiffusionSchedulers, SchedulerMixin


@dataclass
class STORKSchedulerOutput(BaseOutput):
    """
    Output class for the scheduler's `step` function.

    Args:
        prev_sample (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)` for images):
            Computed sample `x_{t-1}` of the previous (lower-noise) timestep; pass it as the model input for the next
            denoising step.
    """

    prev_sample: torch.FloatTensor


def _rkg2_coeff(j: int) -> float:
    """Stage weights of the second-order Runge-Kutta-Gegenbauer stability polynomial (Eq. 3.7 of the paper)."""
    if j == 0:
        return 1.0
    if j == 1:
        return 1.0 / 3.0
    return 4.0 * (j - 1) * (j + 4) / (3.0 * j * (j + 1) * (j + 2) * (j + 3))


class STORKScheduler(SchedulerMixin, ConfigMixin):
    """
    Stabilized orthogonal Runge-Kutta (STORK) scheduler for flow-matching models.

    STORK resolves both failure modes of training-free fast samplers with a single mechanism: a stabilized
    Runge-Kutta-Gegenbauer recurrence takes `num_stages` cheap sub-steps inside a single model evaluation. The recurrence
    stretches the stability region along the negative real axis so large timesteps stay stable (stiffness), and it acts
    directly on the velocity field `v(x, t)` instead of decomposing the ODE into a semi-linear noise/data form, so it
    transfers to flow matching unchanged (structure-dependence). The velocity at each sub-stage is extrapolated from the
    current model output by a low-order Taylor expansion whose derivatives are finite-differenced from the last few
    stored velocity predictions, so the whole `num_stages`-stage advance costs a single function evaluation.

    This implements the recommended second-order Runge-Kutta-Gegenbauer variant with closed-form stage coefficients and
    velocity Taylor orders 1-3, for flow-matching (velocity-prediction) models.

    Proposed in "STORK: Faster Diffusion And Flow Matching Sampling By Resolving Both Stiffness And Structure-Dependence"
    (https://huggingface.co/papers/2505.24210). Adapted from the authors' reference (https://github.com/ZT220501/STORK):
    the flow-matching Runge-Kutta-Gegenbauer path is ported at full fidelity, while the lower-order first-order variant,
    the noise/epsilon diffusion path, the fourth-order ROCK4 variant (which needs precomputed Chebyshev coefficient
    tables), resolution-dependent dynamic shifting, and the alternate Karras/beta/exponential sigma schedules are out of
    scope.

    Args:
        num_train_timesteps (`int`, defaults to 1000):
            The number of diffusion steps the model was trained on; only sets the scale of the public `timesteps` tensor.
        shift (`float`, defaults to 1.0):
            Timestep shift of the flow-matching sigma schedule (`sigma' = shift * sigma / (1 + (shift - 1) * sigma)`).
            `1.0` is the unshifted rectified-flow schedule.
        derivative_order (`int`, defaults to 1):
            Order of the Taylor expansion used to extrapolate the velocity across the sub-stages. `1` uses one past
            prediction, `2` uses two, `3` uses three; higher orders track the velocity field more closely at the cost of
            extra startup steps before the recurrence takes over.
        num_stages (`int`, defaults to 50):
            Number of stabilized Runge-Kutta-Gegenbauer sub-stages taken per model evaluation. More stages widen the
            stability region, allowing fewer (larger) steps; must be `>= 2`.
    """

    _compatibles = [e.name for e in KarrasDiffusionSchedulers]
    order = 1

    @register_to_config
    def __init__(
        self,
        num_train_timesteps: int = 1000,
        shift: float = 1.0,
        derivative_order: int = 1,
        num_stages: int = 50,
    ):
        if derivative_order not in (1, 2, 3):
            raise ValueError(f"`derivative_order` must be 1, 2, or 3; got {derivative_order}.")
        if num_stages < 2:
            raise ValueError(f"`num_stages` must be >= 2; got {num_stages}.")

        self.derivative_order = derivative_order
        self.num_stages = num_stages

        sigmas = torch.linspace(1.0, 0.0, num_train_timesteps + 1, dtype=torch.float32)
        self.sigmas, self.timesteps, self.dt_list = self._build_schedule(sigmas, shift, num_train_timesteps)
        self.init_noise_sigma = 1.0
        self.num_inference_steps = None
        self.velocity_predictions = []
        self._step_index = None

    @staticmethod
    def _build_schedule(sigmas, shift, num_train_timesteps):
        if shift != 1.0:
            sigmas = shift * sigmas / (1.0 + (shift - 1.0) * sigmas)
        timesteps = sigmas[:-1] * num_train_timesteps
        dt_list = sigmas[:-1] - sigmas[1:]
        return sigmas, timesteps, dt_list

    def set_timesteps(self, num_inference_steps: int, device: str | torch.device | None = None) -> None:
        """
        Sets the discrete flow-matching sigmas used for sampling (run before inference).

        Args:
            num_inference_steps (`int`):
                The number of model evaluations used to generate a sample.
            device (`str` or `torch.device`, *optional*):
                The device to move the sigmas and timesteps to.
        """
        if num_inference_steps <= 0:
            raise ValueError(f"`num_inference_steps` must be > 0, got {num_inference_steps}.")
        self.num_inference_steps = num_inference_steps
        sigmas = torch.linspace(1.0, 0.0, num_inference_steps + 1, dtype=torch.float32, device=device)
        self.sigmas, self.timesteps, self.dt_list = self._build_schedule(
            sigmas, float(self.config.shift), self.config.num_train_timesteps
        )
        self.velocity_predictions = []
        self._step_index = None

    def scale_model_input(self, sample: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """Flow-matching velocity models take the sample unscaled."""
        return sample

    def __len__(self) -> int:
        return self.config.num_train_timesteps

    @staticmethod
    def _taylor_extrapolate(order, diff, value, d1, d2, d3):
        """Approximate the velocity a `diff` offset away from the evaluated timestep using its stored derivatives."""
        if order == 1:
            return value + diff * d1
        if order == 2:
            return value + diff * d1 + 0.5 * diff * diff * d2
        return value + diff * d1 + 0.5 * diff * diff * d2 + (diff**3) * d3 / 6.0

    def _rkg2_recurrence(self, model_output, sample, dt, d1, d2, d3):
        """Advance one major step by the second-order Runge-Kutta-Gegenbauer recurrence over `num_stages` sub-stages."""
        s = self.num_stages
        order = self.derivative_order
        first_mu_tilde = 6.0 / ((s + 4) * (s - 1))
        denom = s * s + s - 2
        y_prev2 = sample
        y_prev1 = sample
        y = sample
        for j in range(1, s + 1):
            if j == 1:
                y = y_prev1 - dt * first_mu_tilde * model_output
            else:
                fraction = 4.0 / (3.0 * denom) if j == 2 else ((j - 1) * (j - 1) + (j - 1) - 2) / denom
                mu = (2 * j + 1) * _rkg2_coeff(j) / (j * _rkg2_coeff(j - 1))
                nu = -(j + 1) * _rkg2_coeff(j) / (j * _rkg2_coeff(j - 2))
                mu_tilde = mu * first_mu_tilde
                gamma_tilde = -mu_tilde * (1.0 - j * (j + 1) * _rkg2_coeff(j - 1) / 2.0)
                velocity = self._taylor_extrapolate(order, -fraction * dt, model_output, d1, d2, d3)
                y = (
                    mu * y_prev1
                    + nu * y_prev2
                    + (1.0 - mu - nu) * sample
                    - dt * mu_tilde * velocity
                    - dt * gamma_tilde * model_output
                )
            y_prev2 = y_prev1
            y_prev1 = y
        return y

    def _adams_bashforth_startup(self, model_output, sample, step_index):
        """Two-step Adams-Bashforth startup used until enough velocity predictions are stored for the Taylor order."""
        dt_list = self.dt_list
        return (
            sample
            - 1.5 * model_output * float(dt_list[step_index])
            + 0.5 * self.velocity_predictions[-1] * float(dt_list[step_index - 1])
        )

    def step(
        self,
        model_output: torch.Tensor,
        timestep: int | torch.Tensor,
        sample: torch.Tensor,
        return_dict: bool = True,
    ) -> STORKSchedulerOutput | tuple[torch.Tensor]:
        """
        One STORK (Runge-Kutta-Gegenbauer) step of the flow-matching probability ODE.

        Args:
            model_output (`torch.Tensor`):
                Velocity field `v(x_t, t)` from the model (a single function evaluation).
            timestep (`int` or `torch.Tensor`):
                Current discrete timestep; STORK tracks its position in the schedule internally, so this is only used to
                order the calls.
            sample (`torch.Tensor`):
                Current sample `x_t`.
            return_dict (`bool`, defaults to `True`):
                Whether to return a [`STORKSchedulerOutput`] or a plain tuple.

        Returns:
            [`STORKSchedulerOutput`] or `tuple`: the next sample `x_{t-1}`, cast back to `model_output`'s dtype.
        """
        original_dtype = model_output.dtype
        model_output = model_output.to(torch.float32)
        sample = sample.to(dtype=torch.float32, device=model_output.device)

        if self._step_index is None:
            self._step_index = 0

        dt_list = self.dt_list
        vp = self.velocity_predictions
        step_index = self._step_index
        order = self.derivative_order

        # Startup: the first step seeds the velocity history with a plain Euler flow step.
        if step_index == 0:
            prev_sample = sample - model_output * float(dt_list[0])
            vp.append(model_output)
            self._step_index += 1
            return self._pack(prev_sample, original_dtype, return_dict)

        # Finite-difference the velocity field for the sub-stage Taylor extrapolation. Higher orders need more history;
        # until enough predictions are stored, fall back to a two-step Adams-Bashforth startup step.
        h_prev = float(dt_list[step_index - 1])
        needs_startup = (order == 2 and step_index == 1) or (order == 3 and step_index in (1, 2))
        if needs_startup:
            prev_sample = self._adams_bashforth_startup(model_output, sample, step_index)
            vp.append(model_output)
            self._step_index += 1
            return self._pack(prev_sample, original_dtype, return_dict)

        if order == 1:
            d1 = (vp[-1] - model_output) / h_prev
            d2 = d3 = None
        elif order == 2:
            h2 = float(dt_list[step_index - 2])
            d1 = (-vp[-2] + 4.0 * vp[-1] - 3.0 * model_output) / (2.0 * h_prev)
            d2 = 2.0 / (h_prev * h2 * (h_prev + h2)) * (vp[-2] * h_prev - vp[-1] * (h_prev + h2) + model_output * h2)
            d3 = None
        else:  # order == 3
            h2 = h_prev + float(dt_list[step_index - 2])
            h3 = h2 + float(dt_list[step_index - 3])
            denom = h_prev * h2 * h3
            d1 = (
                (h2 * h3) * (vp[-1] - model_output)
                - (h_prev * h3) * (vp[-2] - model_output)
                + (h_prev * h2) * (vp[-3] - model_output)
            ) / denom
            d2 = (
                2.0
                * (
                    (h2 + h3) * (vp[-1] - model_output)
                    - (h_prev + h3) * (vp[-2] - model_output)
                    + (h_prev + h2) * (vp[-3] - model_output)
                )
                / denom
            )
            d3 = (
                6.0
                * (
                    (h2 - h3) * (vp[-1] - model_output)
                    + (h3 - h_prev) * (vp[-2] - model_output)
                    + (h_prev - h2) * (vp[-3] - model_output)
                )
                / denom
            )

        vp.append(model_output)
        self._step_index += 1
        dt = float(dt_list[step_index])
        prev_sample = self._rkg2_recurrence(model_output, sample, dt, d1, d2, d3)
        return self._pack(prev_sample, original_dtype, return_dict)

    def _pack(self, prev_sample, original_dtype, return_dict):
        prev_sample = prev_sample.to(original_dtype)
        if not return_dict:
            return (prev_sample,)
        return STORKSchedulerOutput(prev_sample=prev_sample)


__all__ = ["STORKScheduler", "STORKSchedulerOutput"]
