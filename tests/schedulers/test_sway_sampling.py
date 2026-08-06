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

import math

import torch

from diffusers import FlowMatchEulerDiscreteScheduler
from diffusers.schedulers.sway_sampling import apply_sway_sampling, sway_sampling


def test_sway_sampling_uniform_when_coef_zero():
    schedule = sway_sampling(num_inference_steps=8, sway_coef=0.0)
    # coef=0 must reproduce the plain uniform descending schedule 1 -> 0
    assert torch.allclose(schedule, torch.linspace(1, 0, 8))


def test_sway_sampling_preserves_endpoints():
    schedule = sway_sampling(num_inference_steps=8, sway_coef=-1.0)
    # F5-TTS transform keeps the endpoints fixed: pure noise (1) and clean data (0)
    assert schedule[0].item() == 1.0
    assert schedule[-1].item() == 0.0
    assert bool(torch.all(schedule <= 1.0)) and bool(torch.all(schedule >= 0.0))


def test_sway_sampling_matches_reference_formula():
    # At coef=-1 the transform reduces to sigma = cos(pi/2 * t); midpoint t=0.5 -> cos(pi/4)
    schedule = sway_sampling(num_inference_steps=5, sway_coef=-1.0)  # t = [0, .25, .5, .75, 1]
    assert math.isclose(schedule[2].item(), math.cos(math.pi / 4), rel_tol=1e-5)


def test_sway_sampling_concentrates_near_noisy_start():
    # coef=-1 places more steps near sigma=1 (pure noise) than the uniform schedule
    sway = sway_sampling(num_inference_steps=20, sway_coef=-1.0)
    uniform = torch.linspace(1, 0, 20)
    assert sway.mean() > uniform.mean()
    assert int((sway > 0.7).sum()) > int((uniform > 0.7).sum())


def test_apply_sway_sampling_sets_scheduler_schedule():
    scheduler = FlowMatchEulerDiscreteScheduler(shift=1.0)
    apply_sway_sampling(scheduler, num_inference_steps=10, sway_coef=-1.0, device="cpu")

    # set_timesteps appends one terminal sigma (0): N+1 sigmas, N timesteps
    assert len(scheduler.sigmas) == 11
    assert len(scheduler.timesteps) == 10
    # spans [sigma_min, sigma_max] with the appended 0 terminal; no double-zero
    assert scheduler.sigmas[0].item() == scheduler.sigma_max
    assert scheduler.sigmas[-1].item() == 0.0
    assert scheduler.sigmas[-2].item() > 0.0


def test_apply_sway_sampling_matches_uniform_at_coef_zero():
    # coef=0 with shift=1.0 must reproduce the scheduler's own uniform schedule
    sway = FlowMatchEulerDiscreteScheduler(shift=1.0)
    apply_sway_sampling(sway, num_inference_steps=10, sway_coef=0.0, device="cpu")

    default = FlowMatchEulerDiscreteScheduler(shift=1.0)
    default.set_timesteps(10, device="cpu")

    assert torch.allclose(sway.sigmas, default.sigmas, atol=1e-3)


def test_apply_sway_sampling_differs_from_default():
    sway = FlowMatchEulerDiscreteScheduler(shift=1.0)
    apply_sway_sampling(sway, num_inference_steps=10, sway_coef=-1.0, device="cpu")

    default = FlowMatchEulerDiscreteScheduler(shift=1.0)
    default.set_timesteps(10, device="cpu")

    assert not torch.allclose(sway.sigmas, default.sigmas)
