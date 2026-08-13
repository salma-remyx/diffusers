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

import torch

from diffusers import PASScheduler, FlowMatchEulerDiscreteScheduler


def _fast_trajectory(scheduler, num_samples, dim, seed=0):
    """Build a random uncorrected flow-matching trajectory and its per-step model directions."""
    torch.manual_seed(seed)
    num_steps = len(scheduler.timesteps)
    sigmas = scheduler.sigmas
    x0 = torch.randn(num_samples, dim)
    directions = torch.randn(num_steps, num_samples, dim) * 0.5
    samples = [x0]
    for i in range(num_steps):
        dt = sigmas[i + 1] - sigmas[i]
        samples.append(samples[-1] + dt * directions[i])
    return x0, directions, torch.stack(samples, dim=0)


def test_uncorrected_equals_flow_match_euler():
    # Without armed PAS coordinates, PASScheduler must reproduce FlowMatchEulerDiscreteScheduler exactly — this is the
    # core guarantee that the correction is a strict superset of the base solver (and the SchedulerOutput contract).
    num_steps = 6
    config = {"num_train_timesteps": 1000, "shift": 1.0}
    pas = PASScheduler(**config)
    euler = FlowMatchEulerDiscreteScheduler(**config)
    pas.set_timesteps(num_steps)
    euler.set_timesteps(num_steps)

    torch.manual_seed(1)
    sample = torch.randn(2, 8)
    for t in pas.timesteps:
        model_output = torch.randn(2, 8)
        pas_out = pas.step(model_output, t, sample).prev_sample
        euler_out = euler.step(model_output, t, sample).prev_sample
        assert torch.allclose(pas_out, euler_out, atol=1e-6)
        sample = pas_out


def test_basis_reparameterizes_euler_direction():
    # The first basis vector is the unit sampling direction, so coordinates [||d||, 0, ...] reproduce d exactly —
    # i.e. the PAS reparameterisation leaves the Euler step untouched until a coordinate is changed.
    scheduler = PASScheduler(num_train_timesteps=1000, shift=1.0, pas_num_basis=4)
    scheduler.set_timesteps(4)
    direction = torch.randn(3, 16)
    x0 = torch.randn(3, 16)
    basis = scheduler._pas_basis(x0, direction.unsqueeze(0), direction)  # [B, D, K]
    norm = direction.norm(dim=1, keepdim=True)
    coordinates = torch.zeros(basis.shape[-1])
    coordinates[0] = 1.0
    reparameterised = torch.einsum("bdk,k->bd", basis, coordinates)
    assert torch.allclose(reparameterised, direction / norm, atol=1e-5)


def test_fit_recovers_coordinates_and_adaptive_selection():
    # Reference = fast trajectory corrected by a known shared coordinate C* at the active steps, uncorrected elsewhere.
    # The closed-form fit must recover C* and only select the active (high-curvature) steps.
    num_steps, num_samples, dim = 4, 6, 16
    scheduler = PASScheduler(num_train_timesteps=1000, shift=1.0, pas_tolerance=1e-5)
    scheduler.set_timesteps(num_steps)
    sigmas = scheduler.sigmas
    x0, directions, fast = _fast_trajectory(scheduler, num_samples, dim)

    active = {1, 3}
    coordinate_star = {}
    reference = [x0.clone()]
    for i in range(num_steps):
        dt = sigmas[i + 1] - sigmas[i]
        if i in active:
            basis = scheduler._pas_basis(fast[0], directions[: i + 1], directions[i])
            c = torch.zeros(basis.shape[-1])
            c[1] = 1.0
            coordinate_star[i] = c
            reference.append(fast[i] + dt * torch.einsum("sdk,k->sd", basis, c))
        else:
            reference.append(fast[i] + dt * directions[i])
    reference = torch.stack(reference, dim=0)

    coordinates = scheduler.fit_pas(directions, fast, reference)

    assert set(coordinates.keys()) == active
    for i in active:
        assert torch.allclose(coordinates[i], coordinate_star[i], atol=1e-4), (i, coordinates[i], coordinate_star[i])


def test_step_applies_armed_correction():
    # With a single corrected step at index 0, the online step() must reproduce the reference exactly there (its
    # trajectory buffer matches the fast trajectory, since no earlier step has deviated).
    num_steps, num_samples, dim = 4, 5, 16
    scheduler = PASScheduler(num_train_timesteps=1000, shift=1.0, pas_tolerance=1e-5)
    scheduler.set_timesteps(num_steps)
    sigmas = scheduler.sigmas
    x0, directions, fast = _fast_trajectory(scheduler, num_samples, dim)

    basis = scheduler._pas_basis(fast[0], directions[:1], directions[0])
    c = torch.zeros(basis.shape[-1])
    c[1] = 1.0
    reference_1 = fast[0] + (sigmas[1] - sigmas[0]) * torch.einsum("sdk,k->sd", basis, c)

    reference = torch.stack(
        [fast[0], reference_1]
        + [fast[i] + (sigmas[i + 1] - sigmas[i]) * directions[i] for i in range(1, num_steps)],
        dim=0,
    )
    coordinates = scheduler.fit_pas(directions, fast, reference)
    assert set(coordinates.keys()) == {0}
    scheduler.set_pas_correction(coordinates)

    scheduler.set_timesteps(num_steps)
    sample = fast[0].clone()
    for i, t in enumerate(scheduler.timesteps):
        out = scheduler.step(directions[i], t, sample).prev_sample
        if i == 0:
            assert torch.allclose(out, reference[1], atol=1e-4)
        sample = out
