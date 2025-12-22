+++
title = "Qwen3 Next Inference Log"
date = 2025-12-30
+++

# Qwen3-Next Inference
This is continuing from the previous log of building a single-batch inference engine for Qwen2.5 models. Now that I've built a simple, non-MoE inference, I want to move onto more challenging kernels. A list of new features that Qwen3-Next:
- MoE kernel
- Multi-Token prediction
- Gated Delta Net Module
- Gated Attention Module
- FP8 Quant (Have to do this otherwise it won't fit on GPU)
- Tensor Parallel GDA / GDN (Will have to do this for long context)

Instead of writing a diary-style, I'm going to write each section a little more detailed because there's more learning to be done here.

## Larger Changes first:
Introducing FP8 means there are two large changes - I need to edit the [Tensor struct](https://github.com/kingsleykimm/qwen2.c/blob/main/qwen2c/utils/tensor.cpp) and the [dtype header](https://github.com/kingsleykimm/qwen2.c/blob/main/qwen2c/utils/dtype.h) files, since each FP8 tensor will have a corresponding scale inv tensor.

I also decided to change the model and tokenizer loading structure a little - instead of having multiple different methods first load in the config.json, then the model weights, then the tokenizer_config.json, it is cleaner to just follow huggingface convention and pass in the model directory to be loaded from.


## Gated Delta Net:

[Paper Link](https://arxiv.org/pdf/2412.06464)

[IMO, Best Non-Paper Resource](https://sustcsonglin.github.io/blog/2024/deltanet-1/)

Ideas of linear attention - originated from the linear transformer 2020 paper, which shows that when excluding normalization and QK activations, the operation is formulated as a linear recurrence. This linear recurrence has an analogous matrix form, demonstrating a dual equivalence between the RNN and Transformer methods of attention. A decay term is added to forget historical information.

Delta networks: References the delta update rule, which dynamically erases the old value associated with an input key with a new value, by interpolating with $\beta \in (0, 1)$, the writing strength. This can be rewritten into another linear recurrent equation, which is a first-order linear recurrence with [generalized Householder transition matrices](https://nhigham.com/2020/09/15/what-is-a-householder-matrix/). 

Gated Delta Nets: Just add a data-dependent gating term on the Householder transition matrix to control state decay. This brings the best of both worlds from gating mechanisms and the delta rule. They also connect the recurrent state updates as a closed form solution to an online learning problem, similar to how Linear Attention, Mamba2 and DeltaNet are also closed-form solutions to OL objectives.

Another interesting connection of the delta rule is to Test-Time Training - the delta rule can be seen as optimizing the online regression objective, $\mathcal{L}(S_t) = \frac 1 2 || S_tk_t-v_t||^2$ through test-time SGD, where $\beta$, which was original the writing intensity, is now a data-dependent, adaptive LR. Then Gated Delta Rules are just adding an adaptive weight decay term $\alpha_t$ to the update.


### Implementation Changes:
- I need to now create a new matmul_cublaslt route that handles lower precision types - since Qwen3-Next's quantization config is dynamic, I need to cast down the bf16 activations into fp8 temporarily for the matmul, with accum_type = bf16 back.
- FP8 Tensor cores actually have a few hard conditions that need to be met, mostly because of the matrix dimensions and strides since it is not guaranteed either of those will be 16 bytes now.
- I also wrote a new cast kernel for __nv_bfloat16 -> __nv_fp8_e4m3, which will be needed in order to cast activations down to fp8 for the matmul, as well as for other blocks in fp8 inference.

## More FP8 GEMM, DeepGEMM:
DeepGEMM seems like the best starting point for learning about FP8 kernels. After reading through the repository and looking at the structure, it's clear that I need to implement a more general matmul kernel that can handle fp8 and Grouped GEMMs. This means I should implement a grouped gemm scheduler as well, similar to the one implemented [here](https://github.com/deepseek-ai/DeepGEMM/blob/main/deep_gemm/include/deep_gemm/common/scheduler.cuh). This is now going into more efficient methods of [warp specialization](https://docs.nvidia.com/cutlass/media/docs/cpp/efficient_gemm.html#hopper-warp-specialization), than what I had before, which was the most rudimentary version of pipelining.

After reading more of DeepGEMM, I've realized that Cutlass can be somewhat restrictive when trying to create more general kernels that can handle different edge cases. What I mean here is the make_tma_copy() method provided by the Cutlass API, which handles the set up of the TMA atom which will be used in the copy() method later. The method requires a compile-time integer, ClusterSize, that describes the multicast dimensions. This isn't great because the cluster dimensions will be fixed throughout the lifetime of a persistent GEMM kernel, but there are cases during a Grouped GEMM where you don't want multicast to turn on. An extreme case would be if we are multicasting along the M dimension, and two CTAs adjacent along this dimension belong to different groups. In this scenario multicast should be somehow 'turned off' on the fly, and each CTA should load in only their chunk. This is technically possible with Cutlass by passing in multiple TMACopy atoms, but what I found cleaner was to forego the TMACopy atoms themselves and try imitating DeepGEMM's utility functions to create CUTensorMap tma descriptors.

For reference, here's Cutlass's core method to create a tma descriptor: [link](https://github.com/NVIDIA/cutlass/blob/d4e16f5d4e70cd95049e3708cbee01205abe43c0/include/cute/atom/copy_traits_sm90_tma.hpp#L923). 

For this part, I decided to mostly port over the code that DeepGEMM had already written - setting up the parameters for the [cuTensorMapEncodeTiled](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TENSOR__MEMORY.html#group__CUDA__TENSOR__MEMORY_1ga7c7d2aaac9e49294304e755e6f341d7) was relatively straightforward, and I don't know if there is too much to change there. Since I was now passing my own Cutensormap objects, like DeepGEMM I needed to create my own dispatch method for TMA copies that could handle 2D/3D and non-multicast.

Later Reads:

[Dynamic Quantization](https://selek.tech/posts/static-vs-dynamic-quantization-in-machine-learning/)