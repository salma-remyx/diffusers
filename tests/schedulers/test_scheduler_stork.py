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

import unittest

import torch

from diffusers import STORKScheduler


class STORKSchedulerTest(unittest.TestCase):
    def get_scheduler(self, **kwargs):
        num_inference_steps = kwargs.pop("num_inference_steps", 8)
        scheduler = STORKScheduler(**kwargs)
        scheduler.set_timesteps(num_inference_steps)
        return scheduler

    def _denoise(self, scheduler, velocity_fn, sample):
        x = sample.clone()
        for t in scheduler.timesteps:
            x = scheduler.step(velocity_fn(x, t), t, x, return_dict=False)[0]
        return x

    def test_set_timesteps(self):
        scheduler = self.get_scheduler()
        scheduler.set_timesteps(16)
        self.assertEqual(scheduler.num_inference_steps, 16)
        self.assertEqual(len(scheduler.timesteps), 16)
        # flow-matching schedule carries one trailing terminal sigma
        self.assertEqual(len(scheduler.sigmas), 17)
        self.assertEqual(len(scheduler.dt_list), 16)

    def test_step_preserves_shape_and_dtype(self):
        scheduler = self.get_scheduler(num_stages=8)
        sample = torch.randn(2, 3, 8, 8, dtype=torch.float16)
        velocity = torch.randn(2, 3, 8, 8, dtype=torch.float16)
        out = scheduler.step(velocity, scheduler.timesteps[0], sample)
        self.assertEqual(out.prev_sample.shape, sample.shape)
        self.assertEqual(out.prev_sample.dtype, torch.float16)

    def test_constant_velocity_matches_exact_flow(self):
        # A constant velocity field makes the flow-matching ODE linear. The Runge-Kutta-Gegenbauer recurrence is
        # consistent, so integrating the full sigma: 1 -> 0 schedule must recover x_0 = x_1 - v exactly, for every
        # Taylor order (higher orders degenerate to the same constant-velocity update).
        torch.manual_seed(0)
        sample = torch.randn(2, 3, 4, 4)
        velocity = torch.randn(2, 3, 4, 4)
        for derivative_order in (1, 2, 3):
            scheduler = self.get_scheduler(derivative_order=derivative_order, num_stages=8)
            x_final = self._denoise(scheduler, lambda x, t: velocity, sample)
            torch.testing.assert_close(x_final.float(), (sample - velocity).float(), rtol=1e-3, atol=1e-3)

    def test_converges_to_data_under_contracting_velocity(self):
        # The linear field v(x) = x - x_data gives the flow-matching ODE dx/dt = x - x_data, whose exact flow contracts
        # by e^{-1} over the sigma: 1 -> 0 sampling schedule. STORK must therefore move the noisy latent toward x_data
        # (stability under a state-dependent velocity) rather than diverge.
        torch.manual_seed(1)
        x_data = torch.randn(1, 3, 4, 4)
        x_noisy = torch.randn(1, 3, 4, 4)
        scheduler = self.get_scheduler(num_inference_steps=10, num_stages=10)
        x_final = self._denoise(scheduler, lambda x, t: x - x_data, x_noisy)
        self.assertLess((x_final - x_data).norm().item(), (x_noisy - x_data).norm().item())

    def test_more_steps_do_not_diverge(self):
        # Fewer NFE means larger steps; the stabilised recurrence should still produce a finite, comparable sample.
        torch.manual_seed(2)
        sample = torch.randn(1, 3, 4, 4)
        velocity = torch.randn(1, 3, 4, 4)
        for num_inference_steps in (4, 8, 16):
            scheduler = self.get_scheduler(num_inference_steps=num_inference_steps, num_stages=12)
            x_final = self._denoise(scheduler, lambda x, t: velocity, sample)
            self.assertTrue(torch.all(torch.isfinite(x_final)))

    def test_return_tuple(self):
        scheduler = self.get_scheduler(num_stages=8)
        sample = torch.randn(1, 3, 4, 4)
        velocity = torch.randn(1, 3, 4, 4)
        out = scheduler.step(velocity, scheduler.timesteps[0], sample, return_dict=False)
        self.assertIsInstance(out, tuple)
