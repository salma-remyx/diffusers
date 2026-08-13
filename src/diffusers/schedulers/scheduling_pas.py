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

r"""PCA-based Adaptive Search (PAS) correction for few-step flow-matching sampling.

This scheduler implements the core mechanism of "Diffusion Sampling Correction via Approximately 10 Parameters"
(Zhang et al., 2024, https://arxiv.org/abs/2411.06503): few-step solvers accumulate a small truncation error that
lives in a low-dimensional subspace of the sampling trajectory. PAS reparameterises each step's sampling direction
onto a small orthonormal basis built from the *current sample's own trajectory* and applies a shared, learned set of
coordinates (the "approximately 10 parameters") to correct it.

Adapted port (Mode 2). The core mechanism is kept at full fidelity:
  - Per-step basis ``U`` of ``num_basis`` (default 4) orthonormal vectors. The first vector is the current sampling
    direction ``d / ||d||``; the rest are the top principal components of the running trajectory
    ``[x_T, d_0, ..., d_i]`` (SVD), orthogonalised against it via Gram-Schmidt.
  - The corrected direction is ``U @ C`` where ``C`` holds one coordinate per basis vector. With ``C`` initialised to
    ``[||d||, 0, ...]`` the step is *identical* to plain flow-matching Euler.
  - The coordinates are shared across all samples (only ``~num_basis`` floats per corrected step are stored).
  - An adaptive search (tolerance ``pas_tolerance`` on the before/after correction gap) selects which high-curvature
    steps actually need correcting.

Auxiliary components substituted for target-native equivalents:
  - Base solver is the repo's rectified-flow Euler step (inherited from ``FlowMatchEulerDiscreteScheduler``) instead of
    the paper's DDIM/DPM-Solver, matching this library's dominant flow-matching stack (Flux, SD3, Cosmos, ...).
  - The paper optimises the coordinates with SGD under an L1 loss against teacher trajectories. Here they are solved in
    closed form as the least-squares optimum (the per-sample-averaged projection of the target direction onto ``U``),
    which is the exact L2 minimiser and needs no bespoke optimiser or training loop.
  - Teacher / calibration trajectory generation is not bundled: the caller supplies paired fast and reference samples
    (a high-NFE run of any scheduler serves as the reference).
"""

from dataclasses import dataclass

import torch

from ..utils import BaseOutput
from .scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler


@dataclass
class PASSchedulerOutput(BaseOutput):
    """
    Output class for the scheduler's `step` function output.

    Args:
        prev_sample (`torch.FloatTensor`):
            Computed sample `(x_{t-1})` of previous timestep. `prev_sample` should be used as next model input in the
            denoising loop.
    """

    prev_sample: torch.FloatTensor


class PASScheduler(FlowMatchEulerDiscreteScheduler):
    """Flow-matching Euler scheduler augmented with a PAS low-rank trajectory correction.

    Behaves exactly like [`FlowMatchEulerDiscreteScheduler`] until correction coordinates are armed with
    [`set_pas_correction`]; armed steps replace the sampling direction with ``U @ C`` where ``U`` is a per-sample PCA
    basis of the trajectory and ``C`` are the shared learned coordinates.

    Args:
        num_train_timesteps (`int`, defaults to 1000):
            The number of diffusion steps the model was trained with.
        shift (`float`, defaults to 1.0):
            Shift value for the timestep schedule. See [`FlowMatchEulerDiscreteScheduler`].
        pas_num_basis (`int`, defaults to 4):
            Number of orthonormal basis vectors per corrected step. The first is always the current sampling direction;
            the remaining ones are principal components of the trajectory. Fewer vectors are used at the early steps
            where the trajectory is too short to fill the basis.
        pas_tolerance (`float`, defaults to 1e-2):
            Adaptive-search tolerance. During [`fit_pas`] a step is corrected only when the mean squared-error gap
            between the uncorrected and corrected step (against the reference) exceeds this value, i.e. when the local
            trajectory curvature is high enough for correction to help.
    """

    def __init__(
        self,
        num_train_timesteps: int = 1000,
        shift: float = 1.0,
        pas_num_basis: int = 4,
        pas_tolerance: float = 1e-2,
    ):
        super().__init__(num_train_timesteps=num_train_timesteps, shift=shift)
        if pas_num_basis < 1:
            raise ValueError("`pas_num_basis` must be at least 1.")
        if pas_tolerance < 0:
            raise ValueError("`pas_tolerance` must be non-negative.")
        self.pas_num_basis = pas_num_basis
        self.pas_tolerance = pas_tolerance
        self.reset_pas_trajectory()
        self._pas_coordinates = {}

    # ------------------------------------------------------------------ public API

    def reset_pas_trajectory(self):
        """Forget the buffered trajectory so the per-sample PCA basis is rebuilt from scratch.

        Call this (or `set_timesteps`, which calls it) before every fresh generation. Armed correction coordinates are
        intentionally left in place.
        """
        self._pas_x0_flat = None
        self._pas_directions = None

    def set_pas_correction(self, coordinates: dict[int, torch.Tensor]):
        """Arm learned PAS coordinates produced by [`fit_pas`].

        Args:
            coordinates (`dict[int, torch.Tensor]`):
                Maps a step index (into the schedule set by `set_timesteps`) to a 1-D coordinate tensor with one entry
                per basis vector at that step. Steps absent from the dict are sampled with plain flow-matching Euler.
        """
        self._pas_coordinates = {int(step): coord.detach().to("cpu") for step, coord in coordinates.items()}

    def set_timesteps(
        self,
        num_inference_steps: int | None = None,
        device: str | torch.device | None = None,
        sigmas: list[float] | None = None,
        mu: float | None = None,
        timesteps: list[float] | None = None,
    ):
        super().set_timesteps(num_inference_steps, device=device, sigmas=sigmas, mu=mu, timesteps=timesteps)
        self.reset_pas_trajectory()

    def step(
        self,
        model_output: torch.FloatTensor,
        timestep: float | torch.FloatTensor,
        sample: torch.FloatTensor,
        return_dict: bool = True,
    ) -> PASSchedulerOutput | tuple:
        """One flow-matching Euler step, applying the PAS correction when coordinates are armed for this step.

        Args:
            model_output (`torch.FloatTensor`):
                The model's predicted velocity (the sampling direction `d_{t_i}`).
            timestep (`float` or `torch.FloatTensor`):
                The current discrete timestep in the diffusion chain. Must be one of `scheduler.timesteps`.
            sample (`torch.FloatTensor`):
                The current sample `x_{t_i}`.
            return_dict (`bool`, defaults to `True`):
                Whether to return a [`PASSchedulerOutput`] or a tuple.

        Returns:
            [`PASSchedulerOutput`] or `tuple`: the corrected previous sample.
        """
        if (
            isinstance(timestep, int)
            or isinstance(timestep, torch.IntTensor)
            or isinstance(timestep, torch.LongTensor)
        ):
            raise ValueError(
                "Passing integer indices (e.g. from `enumerate(timesteps)`) as timesteps to `PASScheduler.step()` is"
                " not supported. Make sure to pass one of the `scheduler.timesteps` as a timestep."
            )

        if self.step_index is None:
            self._init_step_index(timestep)

        step_idx = self.step_index
        sigma = self.sigmas[step_idx]
        sigma_next = self.sigmas[step_idx + 1]
        dt = sigma_next - sigma

        sample = sample.to(torch.float32)
        direction = model_output.to(torch.float32)

        # Buffer the (uncorrected) trajectory direction so the PCA basis spans this sample's own path.
        direction_flat = direction.reshape(direction.shape[0], -1)
        if self._pas_x0_flat is None:
            self._pas_x0_flat = sample.reshape(sample.shape[0], -1).clone()
            self._pas_directions = []
        self._pas_directions.append(direction_flat)

        coordinates = self._pas_coordinates.get(int(step_idx))
        if coordinates is None:
            prev_sample = sample + dt * direction
        else:
            basis = self._pas_basis(
                self._pas_x0_flat, torch.stack(self._pas_directions, dim=0), direction_flat
            )  # [B, D, K]
            corrected_direction_flat = torch.einsum("bdk,k->bd", basis, coordinates.to(basis))
            prev_sample = sample + dt * corrected_direction_flat.reshape_as(model_output)

        self._step_index += 1
        prev_sample = prev_sample.to(model_output.dtype)

        if not return_dict:
            return (prev_sample,)

        return PASSchedulerOutput(prev_sample=prev_sample)

    def fit_pas(
        self,
        fast_directions: torch.Tensor,
        fast_samples: torch.Tensor,
        reference_samples: torch.Tensor,
    ) -> dict[int, torch.Tensor]:
        """Calibrate PAS coordinates from paired fast and reference trajectories (closed form).

        For each candidate step, the shared coordinate vector is the least-squares optimum: the per-sample-averaged
        projection of the target direction ``(x_ref - x_fast) / dt`` onto that step's PCA basis. A step is kept
        (corrected at inference) only when correction reduces the mean squared error against the reference by more than
        `pas_tolerance` — the adaptive search from the paper.

        Args:
            fast_directions (`torch.Tensor` of shape `[N, S, ...]`):
                The uncorrected model output at each of the `N` student steps for `S` calibration samples.
            fast_samples (`torch.Tensor` of shape `[N + 1, S, ...]`):
                The uncorrected sample at the start of each step (indices `0..N-1`) and the final sample (index `N`).
                `fast_samples[0]` is the initial latent `x_T`.
            reference_samples (`torch.Tensor` of shape `[N + 1, S, ...]`):
                The teacher's sample at the matching timesteps (e.g. a high-NFE run of any scheduler). Indexing matches
                `fast_samples`.

        Returns:
            `dict[int, torch.Tensor]`: maps each adaptively-selected step index to its coordinate vector
            (`num_basis_at_step` floats). Pass the result to [`set_pas_correction`].
        """
        if fast_directions.shape[0] + 1 != fast_samples.shape[0]:
            raise ValueError("`fast_samples` must have one more leading entry than `fast_directions`.")
        if fast_samples.shape != reference_samples.shape:
            raise ValueError("`fast_samples` and `reference_samples` must have the same shape.")

        num_steps = fast_directions.shape[0]
        num_samples = fast_directions.shape[1]
        directions = fast_directions.reshape(num_steps, num_samples, -1).to(torch.float32)
        fast = fast_samples.reshape(num_steps + 1, num_samples, -1).to(torch.float32)
        reference = reference_samples.reshape(num_steps + 1, num_samples, -1).to(torch.float32)
        sigmas = self.sigmas.to(directions)

        x0 = fast[0]  # [S, D]
        coordinates: dict[int, torch.Tensor] = {}
        for i in range(num_steps):
            dt = sigmas[i + 1] - sigmas[i]
            basis = self._pas_basis(x0, directions[: i + 1], directions[i])  # [S, D, K]
            target_direction = (reference[i + 1] - fast[i]) / dt  # [S, D]
            # Shared coordinate = mean over samples of the target projected onto each sample's orthonormal basis.
            coordinate = torch.einsum("sdk,sd->sk", basis, target_direction).mean(dim=0)  # [K]

            uncorrected = fast[i] + dt * directions[i]
            corrected = fast[i] + dt * torch.einsum("sdk,k->sd", basis, coordinate)
            loss_before = ((uncorrected - reference[i + 1]) ** 2).mean(dim=1)  # MSE per sample, [S]
            loss_after = ((corrected - reference[i + 1]) ** 2).mean(dim=1)
            if (loss_before - loss_after).mean() > self.pas_tolerance:
                coordinates[i] = coordinate
        return coordinates

    # ------------------------------------------------------------------ internals

    def _pas_basis(
        self,
        x0: torch.Tensor,
        directions: torch.Tensor,
        current: torch.Tensor,
    ) -> torch.Tensor:
        """Build the PAS orthonormal basis for each sample in the batch.

        Args:
            x0 (`torch.Tensor` of shape `[B, D]`): the initial latent `x_T`, flattened.
            directions (`torch.Tensor` of shape `[T, B, D]`): buffered sampling directions `d_0..d_i`, flattened.
            current (`torch.Tensor` of shape `[B, D]`): the current direction `d_i` (equal to `directions[-1]`),
                flattened.

        Returns:
            `torch.Tensor` of shape `[B, D, K]` with `K <= pas_num_basis` orthonormal columns per sample. The first
            column is `current / ||current||`; the rest are principal components of `[x0, d_0..d_i]` orthogonalised
            against it (and each other) via Gram-Schmidt.
        """
        eps = 1e-8
        num_extra = self.pas_num_basis - 1

        unit_current = current / (current.norm(dim=1, keepdim=True) + eps)  # [B, D]
        candidates = [unit_current]

        if num_extra > 0 and directions.shape[0] > 0:
            trajectory = torch.cat([x0.unsqueeze(-1), directions.permute(1, 2, 0)], dim=-1)  # [B, D, T + 1]
            left_singulars = torch.linalg.svd(trajectory, full_matrices=False).U  # [B, D, k]
            for k in range(min(num_extra, left_singulars.shape[-1])):
                candidates.append(left_singulars[:, :, k])

        # Modified Gram-Schmidt: orthogonalise each candidate against the already-fixed basis vectors.
        basis_cols = []
        for candidate in candidates:
            orthogonal = candidate
            for fixed in basis_cols:
                orthogonal = orthogonal - (orthogonal * fixed).sum(dim=1, keepdim=True) * fixed
            norm = orthogonal.norm(dim=1, keepdim=True)
            # A candidate that is already spanned by the basis collapses to ~0; keep a zero column so the coordinate
            # for it simply contributes nothing at inference.
            basis_cols.append(torch.where(norm > eps, orthogonal / (norm + eps), torch.zeros_like(orthogonal)))
        return torch.stack(basis_cols, dim=-1)  # [B, D, K]
