import unittest

import torch

from diffusers import FlowMatchEulerDiscreteScheduler
from diffusers.schedulers.scheduling_flow_match_euler_inverse_slerp import (
    FlowMatchEulerInverseSlerpScheduler,
    FlowMatchEulerInverseSlerpSchedulerOutput,
)


def _slerp_velocity(v_curr, v_next, slerp_t):
    """Reference implementation of the per-position SlerpFlow velocity correction used by the scheduler."""
    dim_to_reduce = 1 if v_curr.ndim == 4 else -1
    norm_curr = torch.norm(v_curr, p=2, dim=dim_to_reduce, keepdim=True)
    norm_next = torch.norm(v_next, p=2, dim=dim_to_reduce, keepdim=True)
    dir_curr = v_curr / (norm_curr + 1e-6)
    dir_next = v_next / (norm_next + 1e-6)
    dot = (dir_curr * dir_next).sum(dim=dim_to_reduce, keepdim=True)
    dot = torch.clamp(dot, -1 + 1e-5, 1 - 1e-5)
    theta = torch.acos(dot)
    sin_theta = torch.sin(theta)
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


class FlowMatchEulerInverseSlerpSchedulerTest(unittest.TestCase):
    def _make_scheduler(self, **kwargs):
        config = {"num_train_timesteps": 1000, "shift": 1.0, "slerp_t": 0.5}
        config.update(kwargs)
        return FlowMatchEulerInverseSlerpScheduler(**config)

    def test_inverse_schedule_is_reverse_of_forward(self):
        # The inverse schedule must walk data -> noise, i.e. be the exact reverse of the forward
        # FlowMatchEulerDiscreteScheduler's noise -> data schedule. This anchors the new inverse scheduler to the
        # existing forward contract it is meant to invert.
        num_steps = 5
        inverse = self._make_scheduler()
        inverse.set_timesteps(num_steps)
        forward = FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000, shift=1.0)
        forward.set_timesteps(num_steps)

        # Inverse timesteps increase (data -> noise), forward timesteps decrease (noise -> data).
        self.assertGreater(inverse.timesteps[-1].item(), inverse.timesteps[0].item())
        self.assertLess(forward.timesteps[-1].item(), forward.timesteps[0].item())
        # And the inverse is exactly the reverse of the forward evaluation points.
        self.assertTrue(torch.allclose(inverse.timesteps.flip(0), forward.timesteps, atol=1e-3))

    def test_step_preserves_shape_and_is_deterministic(self):
        scheduler = self._make_scheduler()
        scheduler.set_timesteps(6)

        generator = torch.Generator().manual_seed(0)
        sample = torch.randn(1, 4, 8, 8, generator=generator)
        velocities = [torch.randn(1, 4, 8, 8, generator=generator) for _ in scheduler.timesteps]

        def run(start):
            current = start.clone()
            for velocity, timestep in zip(velocities, scheduler.timesteps):
                current = scheduler.step(velocity, timestep, current).prev_sample
            return current

        first = run(sample)
        scheduler.set_timesteps(6)  # resets the velocity cache
        second = run(sample)

        self.assertEqual(first.shape, sample.shape)
        self.assertTrue(torch.allclose(first, second))

    def test_first_step_is_euler_then_slerp_corrects_direction(self):
        scheduler = self._make_scheduler()
        scheduler.set_timesteps(4)

        sample = torch.zeros(1, 4, 8, 8)
        v0 = torch.randn(1, 4, 8, 8, generator=torch.Generator().manual_seed(1))
        v1 = torch.randn(1, 4, 8, 8, generator=torch.Generator().manual_seed(2))

        # Step 0: no cached velocity yet -> plain Euler, no Slerp correction.
        out0 = scheduler.step(v0, scheduler.timesteps[0], sample).prev_sample
        dt0 = scheduler.sigmas[1] - scheduler.sigmas[0]
        self.assertTrue(torch.allclose(out0.float(), (sample + dt0 * v0).float(), atol=1e-6))
        self.assertIsNotNone(scheduler._cached_velocity)

        # Step 1: the cached v0 is spherically blended with v1, so the result is the Euler step with the corrected
        # velocity, which differs from a plain Euler step with v1.
        out1 = scheduler.step(v1, scheduler.timesteps[1], out0).prev_sample
        dt1 = scheduler.sigmas[2] - scheduler.sigmas[1]
        corrected = _slerp_velocity(v0.float(), v1.float(), scheduler.config.slerp_t)
        self.assertTrue(torch.allclose(out1.float(), (out0.float() + dt1 * corrected), atol=1e-5))
        self.assertFalse(torch.allclose(out1.float(), (out0.float() + dt1 * v1).float(), atol=1e-4))

    def test_velocity_cache_resets_on_set_timesteps(self):
        scheduler = self._make_scheduler()
        scheduler.set_timesteps(4)
        scheduler.step(torch.randn(1, 4, 8, 8), scheduler.timesteps[0], torch.randn(1, 4, 8, 8))
        self.assertIsNotNone(scheduler._cached_velocity)

        scheduler.set_timesteps(4)
        self.assertIsNone(scheduler._cached_velocity)

    def test_return_tuple_when_not_return_dict(self):
        scheduler = self._make_scheduler()
        scheduler.set_timesteps(3)
        sample = torch.randn(1, 4, 8, 8)
        output = scheduler.step(torch.randn(1, 4, 8, 8), scheduler.timesteps[0], sample, return_dict=False)
        self.assertIsInstance(output, tuple)
        self.assertIsInstance(output[0], torch.Tensor)

        dataclass_out = scheduler.step(torch.randn(1, 4, 8, 8), scheduler.timesteps[1], sample)
        self.assertIsInstance(dataclass_out, FlowMatchEulerInverseSlerpSchedulerOutput)
