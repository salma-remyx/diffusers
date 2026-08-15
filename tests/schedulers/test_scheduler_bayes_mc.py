import unittest

import torch

from diffusers import BayesMCScheduler
from diffusers.schedulers.scheduling_discrete_ddim import DiscreteDDIMScheduler, DiscreteDDIMSchedulerOutput


class BayesMCSchedulerTest(unittest.TestCase):
    def get_scheduler(self, **kwargs):
        config = {"num_inference_steps": 8}
        config.update(kwargs)
        return BayesMCScheduler(**config)

    def test_is_a_discrete_ddim_scheduler(self):
        # The pipeline drives every discrete scheduler through the duck-typed `step` contract of
        # `DiscreteDDIMScheduler`, so the ensemble scheduler must satisfy the same interface.
        scheduler = self.get_scheduler()
        self.assertIsInstance(scheduler, DiscreteDDIMScheduler)
        out = scheduler.step(torch.zeros(1, 4, 10), 0, torch.zeros(1, 4, dtype=torch.long))
        self.assertIsInstance(out, DiscreteDDIMSchedulerOutput)

    def test_invalid_ensemble_size(self):
        with self.assertRaises(ValueError):
            self.get_scheduler(ensemble_size=0)

    def test_set_timesteps(self):
        scheduler = self.get_scheduler()
        scheduler.set_timesteps(16)
        self.assertEqual(scheduler.num_inference_steps, 16)
        self.assertEqual(len(scheduler.timesteps), 16)

    def test_renoise_ensemble_shapes_and_rates(self):
        scheduler = self.get_scheduler(ensemble_size=6)
        scheduler.set_timesteps(8)
        sample = torch.randint(0, 50, (2, 32))
        views, corrupted = scheduler.renoise_ensemble(sample, timestep=0, vocab_size=50)
        self.assertEqual(views.shape, (6, 2, 32))
        self.assertEqual(corrupted.shape, (6, 2, 32))
        # At step 0 the corruption rate is 1 - alpha = 1, so every position is re-corrupted.
        self.assertTrue(bool(corrupted.all()))

        _, corrupted_late = scheduler.renoise_ensemble(sample, timestep=7, vocab_size=50)
        # At the last step alpha = 7/8, so roughly 1/8 of positions are re-corrupted in each view.
        rate = corrupted_late.float().mean().item()
        self.assertLess(rate, 0.35)

    def test_renoise_ensemble_keeps_uncorrupted_tokens(self):
        scheduler = self.get_scheduler(ensemble_size=4)
        scheduler.set_timesteps(8)
        sample = torch.randint(0, 50, (2, 32))
        views, corrupted = scheduler.renoise_ensemble(sample, timestep=7, vocab_size=50)
        # Positions left uncorrupted by a view keep that view's token exactly.
        self.assertTrue(torch.equal(views[~corrupted], sample.unsqueeze(0).expand_as(views)[~corrupted]))

    def test_marginalize_averages_only_corrupted_views(self):
        scheduler = self.get_scheduler(ensemble_size=2)
        # Position 0 is corrupted in both views, position 1 in neither: the corrupted views feed position 0, while
        # position 1 falls back to the plain average over the views.
        corrupted = torch.tensor([[[True, False], [True, False]], [[True, False], [True, False]]])
        logits = torch.zeros(2, 2, 2, 4)
        logits[:, :, 0, 1] = 8.0  # both views put all mass on token 1 at position 0
        logits[0, 0, 1, 2] = 8.0  # view 0 puts mass on token 2 at position 1
        logits[1, 0, 1, 3] = 8.0  # view 1 puts mass on token 3 at position 1
        posterior = scheduler.marginalize(logits, corrupted).exp()
        self.assertAlmostEqual(posterior[0, 0, 1].item(), 1.0, places=5)
        self.assertAlmostEqual(posterior[0, 1, 2].item(), 0.5, places=5)
        self.assertAlmostEqual(posterior[0, 1, 3].item(), 0.5, places=5)

    def test_marginalize_is_normalized(self):
        scheduler = self.get_scheduler(ensemble_size=3)
        corrupted = torch.rand(3, 2, 8) < 0.5
        logits = torch.randn(3, 2, 8, 11)
        posterior = scheduler.marginalize(logits, corrupted).exp()
        self.assertTrue(torch.allclose(posterior.sum(dim=-1), torch.ones(2, 8), atol=1e-5))

    def test_step_accepts_marginalized_logits(self):
        # The pipeline feeds the marginalized log-probabilities straight into the inherited `step`.
        scheduler = self.get_scheduler(ensemble_size=2)
        scheduler.set_timesteps(8)
        sample = torch.randint(0, 20, (2, 16))
        corrupted = torch.rand(2, 2, 16) < 0.5
        ensemble_logits = torch.randn(2, 2, 16, 20)
        logits = scheduler.marginalize(ensemble_logits, corrupted)
        out = scheduler.step(logits, timestep=3, sample=sample, temperature=0.0)
        self.assertEqual(out.prev_sample.shape, sample.shape)

    def test_single_view_ensemble_recovers_discrete_ddim_step(self):
        # With K=1 and a fully corrupted view, the marginalized logits are the plain denoiser log-softmax,
        # so sampling must match `DiscreteDDIMScheduler` fed the same logits under the same RNG.
        sample = torch.randint(0, 20, (2, 16))
        logits = torch.randn(2, 16, 20)
        kwargs = {"temperature": 0.0, "generator": torch.Generator().manual_seed(0)}

        reference = DiscreteDDIMScheduler(num_inference_steps=8)
        reference.set_timesteps(8)
        ref_out = reference.step(logits, timestep=3, sample=sample, **kwargs)

        bayes = self.get_scheduler(ensemble_size=1)
        bayes.set_timesteps(8)
        corrupted = torch.ones(1, 2, 16, dtype=torch.bool)
        marginal = bayes.marginalize(logits.unsqueeze(0), corrupted)
        out = bayes.step(marginal, timestep=3, sample=sample, **kwargs)
        self.assertTrue(torch.equal(out.prev_sample, ref_out.prev_sample))

    def test_full_ensemble_loop_converges_to_consistent_tokens(self):
        # End-to-end: draw views, marginalize, step, and confirm the final step commits the posterior argmax.
        vocab_size = 12
        scheduler = self.get_scheduler(ensemble_size=4)
        scheduler.set_timesteps(4)
        generator = torch.Generator().manual_seed(0)
        canvas = torch.randint(0, vocab_size, (1, 24), generator=generator)
        for step_idx in range(4):
            views, corrupted = scheduler.renoise_ensemble(canvas, step_idx, vocab_size, generator=generator)
            # A denoiser that always predicts token 5 at every position.
            ensemble_logits = torch.zeros(4, 1, 24, vocab_size)
            ensemble_logits[..., 5] = 10.0
            logits = scheduler.marginalize(ensemble_logits, corrupted)
            canvas = scheduler.step(logits, timestep=step_idx, sample=canvas, generator=generator).prev_sample
        self.assertTrue(bool((canvas == 5).all()))
