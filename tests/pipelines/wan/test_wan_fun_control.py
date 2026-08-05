# Copyright 2025 The HuggingFace Team.
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


import pytest
import torch
from PIL import Image
from transformers import AutoConfig, AutoTokenizer, T5EncoderModel

from diffusers import (
    AutoencoderKLWan,
    FlowMatchEulerDiscreteScheduler,
    WanFunControlPipeline,
    WanTransformer3DModel,
)

from ...testing_utils import assert_tensors_close, torch_device
from ..testing_utils import BasePipelineTesterConfig, MemoryTesterMixin, PipelineTesterMixin


class WanFunControlPipelineTesterConfig(BasePipelineTesterConfig):
    pipeline_class = WanFunControlPipeline
    required_input_params_in_call_signature = frozenset(
        ["prompt", "negative_prompt", "height", "width", "guidance_scale", "prompt_embeds", "negative_prompt_embeds"]
    )
    batch_input_params = frozenset(["prompt"])
    # Wan Fun-Control is a video pipeline: it exposes `num_videos_per_prompt`, not the base default
    # `num_images_per_prompt`.
    optional_input_params = frozenset(
        ["num_inference_steps", "num_videos_per_prompt", "generator", "latents", "output_type", "return_dict"]
    )

    def get_dummy_components(self):
        torch.manual_seed(0)
        vae = AutoencoderKLWan(
            base_dim=3,
            z_dim=16,
            dim_mult=[1, 1, 1, 1],
            num_res_blocks=1,
            temperal_downsample=[False, True, True],
        )

        torch.manual_seed(0)
        scheduler = FlowMatchEulerDiscreteScheduler(shift=7.0)
        config = AutoConfig.from_pretrained("hf-internal-testing/tiny-random-t5")
        text_encoder = T5EncoderModel(config)
        tokenizer = AutoTokenizer.from_pretrained("hf-internal-testing/tiny-random-t5")

        torch.manual_seed(0)
        # The transformer takes the noisy latents concatenated, channel-wise, with the VAE-encoded control latents,
        # so `in_channels` is `2 * z_dim` and only the `z_dim` noise channels are denoised back out.
        transformer = WanTransformer3DModel(
            patch_size=(1, 2, 2),
            num_attention_heads=2,
            attention_head_dim=12,
            in_channels=32,
            out_channels=16,
            text_dim=32,
            freq_dim=256,
            ffn_dim=32,
            num_layers=3,
            cross_attn_norm=True,
            qk_norm="rms_norm_across_heads",
            rope_max_seq_len=32,
        )

        return {
            "transformer": transformer,
            "vae": vae,
            "scheduler": scheduler,
            "text_encoder": text_encoder,
            "tokenizer": tokenizer,
            "transformer_2": None,
        }

    def get_dummy_inputs(self):
        num_frames = 17
        height = 16
        width = 16

        control_video = [Image.new("RGB", (height, width))] * num_frames

        return {
            "prompt": "dance monkey",
            "control_video": control_video,
            "negative_prompt": "negative",
            "generator": self.get_generator(0),
            "num_inference_steps": 2,
            "guidance_scale": 6.0,
            "height": 16,
            "width": 16,
            "num_frames": num_frames,
            "max_sequence_length": 16,
            # Request torch outputs so tests compare torch tensors directly (see `BasePipelineTesterConfig`).
            "output_type": "pt",
        }


class TestWanFunControlPipeline(WanFunControlPipelineTesterConfig, PipelineTesterMixin):
    @pytest.mark.skip(reason="Batching is not yet supported with this pipeline")
    def test_inference_batch_consistent(self):
        pass

    @pytest.mark.skip(reason="Batching is not yet supported with this pipeline")
    def test_inference_batch_single_identical(self):
        pass

    def test_inference(self):
        # Run on CPU: the expected slice below is CPU-specific.
        pipe = self.get_pipeline()

        inputs = self.get_dummy_inputs()
        video = pipe(**inputs).frames
        generated_video = video[0]
        assert generated_video.shape == (17, 3, 16, 16)

        # fmt: off
        expected_slice = torch.tensor([0.45243, 0.45225, 0.44918, 0.45346, 0.45193, 0.45311, 0.45428, 0.45346, 0.53023, 0.53449, 0.52846, 0.49807, 0.52182, 0.52256, 0.52594, 0.51535])
        # fmt: on

        generated_slice = generated_video.flatten()
        generated_slice = torch.cat([generated_slice[:8], generated_slice[-8:]])
        assert torch.allclose(generated_slice, expected_slice, atol=1e-3)

    def test_control_video_is_required(self):
        # `control_video` is the control signal this pipeline conditions on; omitting it must raise rather than
        # silently generating an unconditioned video.
        pipe = self.get_pipeline()

        inputs = self.get_dummy_inputs()
        inputs.pop("control_video")
        with pytest.raises(ValueError):
            pipe(**inputs)

    def test_save_load_optional_components(self, tmp_path, expected_max_difference=1e-4):
        # `_optional_components` lists both `transformer` and `transformer_2`, but only `transformer_2` is optional
        # for the single-stage default. The base test nulls every optional component, which would drop the required
        # `transformer` and leave no denoiser, so restrict this to `transformer_2`.
        pipe = self.get_pipeline().to(torch_device)
        pipe.transformer_2 = None

        inputs = self.get_dummy_inputs()
        output = pipe(**inputs)[0]

        pipe.save_pretrained(tmp_path, safe_serialization=False)
        pipe_loaded = self.pipeline_class.from_pretrained(tmp_path)
        pipe_loaded.to(torch_device)
        pipe_loaded.set_progress_bar_config(disable=None)

        assert pipe_loaded.transformer_2 is None, "`transformer_2` did not stay set to None after loading."

        inputs = self.get_dummy_inputs()
        output_loaded = pipe_loaded(**inputs)[0]

        assert_tensors_close(
            output_loaded,
            output,
            atol=expected_max_difference,
            msg="Output changed after dropping the optional component.",
        )


class TestWanFunControlPipelineMemory(WanFunControlPipelineTesterConfig, MemoryTesterMixin):
    pass
