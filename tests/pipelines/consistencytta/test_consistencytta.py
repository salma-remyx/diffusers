# coding=utf-8
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

import numpy as np
import torch
from transformers import (
    SpeechT5HifiGan,
    SpeechT5HifiGanConfig,
    T5Config,
    T5EncoderModel,
    T5Tokenizer,
)

from diffusers import (
    AutoencoderKL,
    CMStochasticIterativeScheduler,
    ConsistencyTTAPipeline,
    UNet2DConditionModel,
)

from ...testing_utils import enable_full_determinism, torch_device
from ..pipeline_params import TEXT_TO_AUDIO_BATCH_PARAMS, TEXT_TO_AUDIO_PARAMS
from ..test_pipelines_common import PipelineTesterMixin


enable_full_determinism()


class ConsistencyTTAPipelineFastTests(PipelineTesterMixin, unittest.TestCase):
    pipeline_class = ConsistencyTTAPipeline
    params = TEXT_TO_AUDIO_PARAMS
    batch_params = TEXT_TO_AUDIO_BATCH_PARAMS
    required_optional_params = frozenset(
        [
            "num_inference_steps",
            "num_waveforms_per_prompt",
            "generator",
            "latents",
            "output_type",
            "return_dict",
            "callback",
            "callback_steps",
        ]
    )

    def get_dummy_components(self):
        torch.manual_seed(0)
        unet = UNet2DConditionModel(
            block_out_channels=(8, 16),
            layers_per_block=1,
            norm_num_groups=8,
            sample_size=32,
            in_channels=4,
            out_channels=4,
            down_block_types=("DownBlock2D", "CrossAttnDownBlock2D"),
            up_block_types=("CrossAttnUpBlock2D", "UpBlock2D"),
            cross_attention_dim=32,
        )
        scheduler = CMStochasticIterativeScheduler(num_train_timesteps=40)
        torch.manual_seed(0)
        vae = AutoencoderKL(
            block_out_channels=[8, 16],
            in_channels=1,
            out_channels=1,
            norm_num_groups=8,
            down_block_types=["DownEncoderBlock2D", "DownEncoderBlock2D"],
            up_block_types=["UpDecoderBlock2D", "UpDecoderBlock2D"],
            latent_channels=4,
        )
        torch.manual_seed(0)
        text_encoder_config = T5Config(
            vocab_size=32100,
            d_model=32,
            d_ff=37,
            d_kv=8,
            num_heads=1,
            num_layers=1,
        )
        text_encoder = T5EncoderModel(text_encoder_config)
        tokenizer = T5Tokenizer.from_pretrained("hf-internal-testing/tiny-random-T5Model", model_max_length=77)

        vocoder_config = SpeechT5HifiGanConfig(
            model_in_dim=8,
            sampling_rate=16000,
            upsample_initial_channel=16,
            upsample_rates=[2, 2],
            upsample_kernel_sizes=[4, 4],
            resblock_kernel_sizes=[3, 7],
            resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5]],
            normalize_before=False,
        )
        vocoder = SpeechT5HifiGan(vocoder_config)

        components = {
            "unet": unet,
            "scheduler": scheduler,
            "vae": vae,
            "text_encoder": text_encoder,
            "tokenizer": tokenizer,
            "vocoder": vocoder,
        }
        return components

    def get_dummy_inputs(self, device, seed=0):
        if str(device).startswith("mps"):
            generator = torch.manual_seed(seed)
        else:
            generator = torch.Generator(device=device).manual_seed(seed)
        inputs = {
            "prompt": "A hammer hitting a wooden surface",
            "generator": generator,
            "num_inference_steps": 1,
            "guidance_scale": 1.0,
        }
        return inputs

    def test_consistencytta_single_step(self):
        device = "cpu"  # ensure determinism for the device-dependent torch.Generator

        components = self.get_dummy_components()
        pipe = ConsistencyTTAPipeline(**components)
        pipe = pipe.to(torch_device)
        pipe.set_progress_bar_config(disable=None)

        # num_inference_steps=1 is the whole point of consistency distillation
        inputs = self.get_dummy_inputs(device)
        output = pipe(**inputs)
        audio = output.audios[0]

        assert audio.ndim == 1
        assert len(audio) == 256

    def test_consistencytta_multistep(self):
        device = "cpu"  # ensure determinism for the device-dependent torch.Generator

        components = self.get_dummy_components()
        pipe = ConsistencyTTAPipeline(**components)
        pipe = pipe.to(torch_device)
        pipe.set_progress_bar_config(disable=None)

        # multistep sampling still produces a fixed-length waveform
        inputs = self.get_dummy_inputs(device)
        inputs["num_inference_steps"] = 4
        output = pipe(**inputs)
        audio = output.audios[0]

        assert audio.ndim == 1
        assert len(audio) == 256

    def test_consistencytta_default_num_inference_steps(self):
        components = self.get_dummy_components()
        pipe = ConsistencyTTAPipeline(**components)
        pipe = pipe.to(torch_device)
        pipe.set_progress_bar_config(disable=None)

        # the default call collapses denoising to a single step
        output = pipe("A hammer hitting a wooden surface")
        audio = output.audios

        assert audio.shape == (1, 256)

    def test_consistencytta_prompt_embeds(self):
        components = self.get_dummy_components()
        pipe = ConsistencyTTAPipeline(**components)
        pipe = pipe.to(torch_device)
        pipe.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_inputs(torch_device)
        inputs["prompt"] = 3 * [inputs["prompt"]]

        # forward with text prompts
        output = pipe(**inputs)
        audio_1 = output.audios[0]

        inputs = self.get_dummy_inputs(torch_device)
        prompt = 3 * [inputs.pop("prompt")]

        text_inputs = pipe.tokenizer(
            prompt,
            padding="max_length",
            max_length=pipe.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        prompt_embeds = pipe.text_encoder(text_inputs.input_ids.to(torch_device))[0]
        inputs["prompt_embeds"] = prompt_embeds

        # forward with pre-computed text embeddings
        output = pipe(**inputs)
        audio_2 = output.audios[0]

        assert np.abs(audio_1 - audio_2).max() < 1e-3

    def test_consistencytta_negative_prompt(self):
        components = self.get_dummy_components()
        pipe = ConsistencyTTAPipeline(**components)
        pipe = pipe.to(torch_device)
        pipe.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_inputs(torch_device)
        inputs["guidance_scale"] = 3.0
        inputs["negative_prompt"] = "egg cracking"

        output = pipe(**inputs)
        audio = output.audios[0]

        assert audio.ndim == 1
        assert len(audio) == 256

    def test_consistencytta_num_waveforms_per_prompt(self):
        device = "cpu"  # ensure determinism for the device-dependent torch.Generator
        components = self.get_dummy_components()
        pipe = ConsistencyTTAPipeline(**components)
        pipe = pipe.to(device)
        pipe.set_progress_bar_config(disable=None)

        prompt = "A hammer hitting a wooden surface"

        # default: one waveform per prompt
        audios = pipe(prompt, num_inference_steps=1).audios
        assert audios.shape == (1, 256)

        # num_waveforms_per_prompt for a batch of prompts
        batch_size = 2
        num_waveforms_per_prompt = 2
        audios = pipe(
            [prompt] * batch_size, num_inference_steps=1, num_waveforms_per_prompt=num_waveforms_per_prompt
        ).audios
        assert audios.shape == (batch_size * num_waveforms_per_prompt, 256)

    def test_consistencytta_audio_length_in_s(self):
        device = "cpu"  # ensure determinism for the device-dependent torch.Generator
        components = self.get_dummy_components()
        pipe = ConsistencyTTAPipeline(**components)
        pipe = pipe.to(torch_device)
        pipe.set_progress_bar_config(disable=None)
        vocoder_sampling_rate = pipe.vocoder.config.sampling_rate

        inputs = self.get_dummy_inputs(device)
        output = pipe(audio_length_in_s=0.016, **inputs)
        audio = output.audios[0]
        assert len(audio) / vocoder_sampling_rate == 0.016

        output = pipe(audio_length_in_s=0.032, **inputs)
        audio = output.audios[0]
        assert len(audio) / vocoder_sampling_rate == 0.032

    def test_attention_slicing_forward_pass(self):
        self._test_attention_slicing_forward_pass(test_mean_pixel_difference=False)

    # `encode_prompt` returns a single `prompt_embeds` tensor (no separate negative/generated embeddings),
    # so the shared isolation harness — which zips `encode_prompt`'s return names against a returned tuple —
    # does not apply. Same as `StableAudioPipeline`.
    def test_encode_prompt_works_in_isolation(self):
        pass

    @unittest.skip("Not supported yet because the `vocoder` is not offloaded.")
    def test_model_cpu_offload_forward_pass(self):
        pass

    @unittest.skip("Not supported yet because the `vocoder` is not offloaded.")
    def test_sequential_cpu_offload_forward_pass(self):
        pass

    @unittest.skip("Not supported yet because the `vocoder` is not offloaded.")
    def test_cpu_offload_forward_pass_twice(self):
        pass

    @unittest.skip("Not supported yet because the `vocoder` is not offloaded.")
    def test_sequential_offload_forward_pass_twice(self):
        pass
