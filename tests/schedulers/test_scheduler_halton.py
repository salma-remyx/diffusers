import inspect
import unittest

import torch

from diffusers import HaltonScheduler
from diffusers.pipelines.diffusion_gemma import DiffusionGemmaPipeline


class HaltonSchedulerTest(unittest.TestCase):
    def get_scheduler(self, **kwargs):
        config = {"num_inference_steps": 8}
        config.update(kwargs)
        return HaltonScheduler(**config)

    def test_set_timesteps(self):
        scheduler = self.get_scheduler()
        scheduler.set_timesteps(16)
        self.assertEqual(scheduler.num_inference_steps, 16)
        self.assertEqual(len(scheduler.timesteps), 16)

    def test_set_timesteps_invalid(self):
        scheduler = self.get_scheduler()
        with self.assertRaises(ValueError):
            scheduler.set_timesteps(0)

    def test_position_ranks_are_a_low_discrepancy_permutation(self):
        # The ranks are a permutation of range(block_length) whose early entries alternate between the grid halves,
        # rather than clustering like a raster scan would.
        ranks = HaltonScheduler.build_position_ranks(256)
        order = ranks.argsort()
        self.assertTrue(torch.equal(ranks.sort().values, torch.arange(256)))
        self.assertFalse(torch.equal(order, torch.arange(256)))
        self.assertTrue(((order[:16] % 16) < 8).float().mean() > 0.2)

    def test_selection_ignores_confidence(self):
        # The core of the paper: positions committed are fixed by the Halton order, not by where the model is confident.
        scheduler = self.get_scheduler(num_inference_steps=8)
        sample = torch.randint(0, 100, (1, 256))
        logits = torch.randn(1, 256, 100, generator=torch.Generator().manual_seed(0))
        confident_logits = torch.zeros(1, 256, 100)
        confident_logits[0, :200, 0] = 1e6  # the first 200 positions are near-certain

        plain = scheduler.step(logits, timestep=0, sample=sample, return_dict=True)
        scheduler.set_timesteps(8)
        confident = scheduler.step(confident_logits, timestep=0, sample=sample, return_dict=True)
        self.assertTrue(torch.equal(plain.transfer_index, confident.transfer_index))
        # ...and it is not "commit the confident prefix".
        self.assertLess(confident.transfer_index[0, :200].sum().item(), 200)

    def test_quota_schedule_commits_every_token_over_the_run(self):
        block_length = 256
        scheduler = self.get_scheduler(num_inference_steps=32)
        scheduler.set_timesteps(32)
        sample = torch.randint(0, 100, (2, block_length))
        logits = torch.randn(2, block_length, 100)

        committed = torch.zeros_like(sample, dtype=torch.bool)
        for step_idx in range(32):
            out = scheduler.step(logits, timestep=step_idx, sample=sample, return_dict=True)
            self.assertFalse((out.transfer_index & committed).any())  # each position committed at most once
            committed |= out.transfer_index
            self.assertGreaterEqual(int(out.transfer_index.sum()) // 2, 1)  # at least one token per row per step
        self.assertTrue(bool(committed.all()))  # the whole block is committed by the last step

    def test_committed_positions_keep_their_token_across_steps(self):
        scheduler = self.get_scheduler(num_inference_steps=4)
        scheduler.set_timesteps(4)
        sample = torch.randint(0, 100, (2, 64))
        logits = torch.randn(2, 64, 100, generator=torch.Generator().manual_seed(0))

        first = scheduler.step(logits, timestep=0, sample=sample, return_dict=True)
        first_committed = first.transfer_index
        self.assertTrue(torch.equal(first.prev_sample[first_committed], first.sampled_tokens[first_committed]))
        second = scheduler.step(logits, timestep=1, sample=first.prev_sample, return_dict=True)
        self.assertTrue(torch.equal(second.prev_sample[first_committed], first.prev_sample[first_committed]))

    def test_mask_mode_leaves_mask_tokens_elsewhere(self):
        scheduler = self.get_scheduler(num_inference_steps=4)
        scheduler.set_timesteps(4)
        sample = torch.full((2, 64), 7, dtype=torch.long)  # 7 is the mask token
        logits = torch.randn(2, 64, 100, generator=torch.Generator().manual_seed(0))
        out = scheduler.step(logits, timestep=0, sample=sample, mask_token_id=7, return_dict=True)
        self.assertTrue(torch.equal(out.prev_sample[~out.transfer_index], sample[~out.transfer_index]))

    def test_randomize_gives_rows_different_orders(self):
        scheduler = self.get_scheduler(num_inference_steps=4, randomize=True)
        scheduler.set_timesteps(4)
        sample = torch.randint(0, 100, (16, 64))
        logits = torch.zeros(16, 64, 100)
        out = scheduler.step(logits, timestep=0, sample=sample, generator=torch.Generator().manual_seed(0))
        rows = {out.transfer_index[i].tolist() for i in range(16)}
        self.assertGreater(len(rows), 1)
        per_row = out.transfer_index[0].sum().item()
        self.assertTrue(all(int(r.sum()) == per_row for r in out.transfer_index))

    def test_step_output_shapes(self):
        scheduler = self.get_scheduler()
        sample = torch.randint(0, 100, (3, 16))
        logits = torch.randn(3, 16, 100)
        out = scheduler.step(logits, timestep=0, sample=sample)
        self.assertEqual(out.prev_sample.shape, sample.shape)
        self.assertEqual(out.transfer_index.shape, sample.shape)
        self.assertEqual(out.sampled_tokens.shape, sample.shape)
        self.assertEqual(out.sampled_probs.shape, sample.shape)
        self.assertEqual(out.pred_logits.shape, logits.shape)

    def test_drives_diffusion_gemma_pipeline(self):
        # The pipeline is scheduler-agnostic: it duck-types on `step(model_output, timestep, sample, mask_token_id=,
        # temperature=, generator=)` and on `set_timesteps(num_inference_steps, device=)`, so the Halton scheduler
        # plugs in next to BlockRefinement/DiscreteDDIM/EntropyBound without any pipeline change.
        accepted = inspect.signature(DiffusionGemmaPipeline.__init__).parameters["scheduler"].annotation
        self.assertIn("HaltonScheduler", accepted)

        canvas_length = 64
        vocab_size = 100
        scheduler = self.get_scheduler(num_inference_steps=8)
        canvas = torch.randint(0, vocab_size, (2, canvas_length))
        logits = torch.randn(2, canvas_length, vocab_size, generator=torch.Generator().manual_seed(0))
        generator = torch.Generator().manual_seed(0)

        step_param_names = set(inspect.signature(scheduler.step).parameters)
        self.assertTrue({"mask_token_id", "generator", "return_dict"} <= step_param_names)
        set_timesteps_kwargs = {"device": canvas.device}
        if "block_length" in inspect.signature(scheduler.set_timesteps).parameters:
            set_timesteps_kwargs["block_length"] = canvas_length
        scheduler.set_timesteps(8, **set_timesteps_kwargs)
        self.assertEqual(scheduler.num_inference_steps, 8)

        for step_idx in range(8):
            step_kwargs = {"mask_token_id": None, "generator": generator}
            step_kwargs = {k: v for k, v in step_kwargs.items() if k in step_param_names}
            out = scheduler.step(
                model_output=logits, timestep=step_idx, sample=canvas, return_dict=True, **step_kwargs
            )
            canvas = out.prev_sample
            self_conditioning_logits = out.pred_logits
        self.assertTrue(torch.logical_and(canvas >= 0, canvas < vocab_size).all())
        self.assertEqual(self_conditioning_logits.shape, logits.shape)
