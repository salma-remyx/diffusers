<!--Copyright 2026 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# BayesMCScheduler

The `BayesMCScheduler` adds Monte Carlo marginalization to discrete DDIM sampling, following
[Bayesian Discrete Diffusion Beats Autoregressive Perplexity](https://huggingface.co/papers/2507.07586). The expected
denoiser output under the forward corruption distribution recovers the exact posterior over clean tokens, so averaging
the denoiser over `ensemble_size` independent re-corruptions of the sequence estimates that posterior at rate
O(1/sqrt(K)) with no extra training. Only positions a view actually corrupted contribute to that position's average.

Each denoising step runs `ensemble_size` denoiser forwards instead of one; `ensemble_size=1` recovers plain
[`DiscreteDDIMScheduler`] sampling. This scheduler is used by [`DiffusionGemmaPipeline`].

## BayesMCScheduler
[[autodoc]] BayesMCScheduler
    - renoise_ensemble
    - marginalize
