+++
title = "The DeepGEMM Scheduler Struct"
date = 2025-12-30
+++

## Intro

While working on implementing the forward pass for the [Qwen3-Next-FP8](https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Thinking-FP8), I needed to learn about how to write efficient FP8 kernels, and DeepGEMM is a fantastic learning source for this. Looking through the code, I noticed a common class that kept appearing in all the GEMMs - the [Scheduler struct](https://github.com/deepseek-ai/DeepGEMM/blob/main/deep_gemm/include/deep_gemm/common/scheduler.cuh). 

This is the real workhorse behind the entire library, as it enables persistent kernel scheduling across the entire CTA grid. Persistent scheduling treats each SM as the unit of work instead of the GEMM tile. In cases where the problem size of a GEMM is too large (common in small MN, large K problems, or Grouped GEMMs), one iteration through all the SMs won't be enough to complete the problem, so computation is done in waves of tiles. This is where an explicit hardware-optimized tile scheduler can provide performance gains, and is almost necessary.

This [Colfax tutorial](https://research.colfax-intl.com/cutlass-tutorial-persistent-kernels-and-stream-k/) is a great read about persistent kernels, and as always, they provide intuitive visuals and some example code which is helpful.

## The Scheduler Object

Important template parameters / objects:
- **GemmType**: chosen from batched or group configurations
- **Problem shape**: BLOCK_M, BLOCK_N, numGroups, numSMs,
- **Num1DBlocksPerGroup**: Used in Threadblock Swizzling, a technique to increase L2 cache hit rate
- **Grouped GEMM parameters**: Grouped Layout, current_group_idx, current_m_cumsum
- **is_peer_cta_alive**: public variable used in GEMMs to deal with boundary / odd dimension cases of multicast

Constructor: Based on the GemmType, the parameters are initialized in different ways. DeepGEMM provides five different GemmTypes: Normal, Batched, MGroupedContiguous, MGroupedMasked, KGroupedContiguous. I'm going to leave off KGroupedContiguous for now. Normal and Batched are self-explanatory - the MGrouped* types are new to me and are actually used for [sparse MoEs](https://huggingface.co/blog/moe).

For a given tensor of shape (batch_size, seq_len, hidden_dim), which is fed into the MoE layer, each token gets sent to a fixed number of experts. Usually though consecutive tokens are not sent to the same experts, so each row will require a different expert weight matrix. Thus we need to group a set of token rows that share an expert together to take advantage of GEMMs. 

MGroupedContiguous describes one way of doing this: it permutes the tensor in-place so that a set of tokens with the same expert is contiguous, and the entire tensor is still contiguous. MGroupedMasked instead constructs 'slabs' per expert, with the rows filled by any tokens that are routed to the ith expert, without contiguous guarantees. Both tensors use pad their M-dimensions (num_tokens) to a multiple of 64 to align with WGMMA specs.

Another key difference between the MGrouped* variants is how their grouped_layouts are stored. Grouped_layouts is a uint32_t array that contains the expert values assigned to each M row. 

For MGroupedContiguous, the rows are stored in a contiguous matrix, with some padding values in between the end and start of different expert chunks. Thus it has the shape (num_m_blocks, BLOCK_M) - for a given grouped_layout[m_block_idx][out_row], the value will either be the index of an expert or -1, for padding value.

For MGroupedMasked, since the matrices are not contiguous, grouped_layout is simpler, with length (kNumGroups), and each index just holds the number of rows this group/expert 'owns', since we don't need to worry about boundary conditions.


### Methods:
I am going to first start with the methods they use to perform thread-block swizzling, not to be confused with the [tensor swizzling](https://docs.nvidia.com/cuda/parallel-thread-execution/#tensor-swizzling-modes) commonly done to avoid shared memory conflicts. Thread-block swizzling is a strategy to increase L2 cache rates when loading in either the M or N block tiles. If we swizzle across the N dimension of the output tile, with a swizzle size of 8, this means that for the current swizzle block, (num_m_blocks, SWIZZLE_SIZE),  only up to 8 of the N blocks are fetched. By keeping the swizzle size reasonable, these N blocks will be kept in L2 cache across the CTAs in the swizzle block. 

They implement a small heuristic search method:
```cpp
template <GemmType kGemmType, uint32_t BLOCK_M, uint32_t BLOCK_N,
          uint32_t kNumSMs, bool kIsMulticastOnA>
static constexpr uint32_t get_num_1d_blocks_per_group()
```
Where the candidates for the swizzle size are 8 or 16, and numSms is the number of sms used by the scheduler. kIsMulticastOnA is used to determine which direction the swizzle should be - the non-multicast direction. I noticed here that the scheduler doesn't support 2D multicast, which is a little rigid, but it makes sense for most decode settings where one of M/N will be comparably larger to the other.

The method above is used to calculate the template variable kNum1DBlocksPerGroup for Scheduler, and used in the method:
```cpp
__device__ __forceinline__ void
   get_swizzled_block_idx(const uint32_t &block_idx, 
uint32_t &m_block_idx, uint32_t &n_block_idx) 
```

This takes in the current block_idx, which is a 1D index into the total number of blocks scheduled over the lifetime of a persistent kernel. It then identifies the direction of the swizzling, and the group_idx variable, which is the ith chunk when chunking the swizzle dimension by swizzle size. Afterwards, m_block_idx and n_block_idx are modified based off the dimensions of the swizzle chunk, and then it returns. It also provides a simple check for multicast settings where the tail blocks are odd - it then adaptively truncates the group of tail blocks by one, and then sets up a new group block for the last, odd block out.


```cpp
__device__ __forceinline__ bool get_next_block(
   uint32_t &m_block_idx, uint32_t &n_block_idx)
```
This is the entrypoint to the tile-scheduler - every CTA/SM calls this to get assigned their next block in the problem space, which is first represented by the 1D index next_block_idx. Slightly different logic for each of the GemmTypes:

MGroupedMasked:
- Iterate over each group until we get through all of them, or find the kth Group where next_block_idx is less than the cumulative sum of m_blocks so far, found through grouped_layout. Then return the threadblock swizzled index.

Batched:
- group_idx = batch_idx, use that to find the local block_idx in the current batch. Then performs a small check on the Multicast dimension to figure out how to determine the major-axis of the tiles.

MGroupedContiguous: 
- The num_m_blocks is already equivalent across each group because of padding, num_n_block stays the same, so there is no need to change anything. Return by the threadblock swizzled index.


```cpp
template <bool kWithGroupOffset, IndexType kIndexType = IndexType::MN>
__device__ __forceinline__ uint32_t
  get_global_idx(const uint32_t shape_dim, const uint32_t block_size,
                 const uint32_t &block_idx, const uint32_t
                 &m_block_idx = 0) 
```

This method is pretty straightforward - it takes in the given shape_dim, i.e the problem size M, block_size : block_M, and then an important template kWithGroupOffset as well, and outputs the global_idx along the M/N/K dimension to load in from GMEM.
The kWithGroupOffset variable, when set to true will advance to the appropriate group slice, for either M (activations) or N (weights).


```cpp
__device__ __forceinline__ bool
  is_tma_multicast_valid(const uint32_t &m_block_idx)
```
This is a short helper method that is really only used for the MGroupedContiguous case. What it does is check for boundary conditions on the m_block_idx, since it could be the case that two M-adjacent CTAs, which are participating in multicast together could be in different groups. Then multicast should not be used

```cpp
__device__ __forceinline__ bool
  is_computation_valid(const uint32_t &m_block_idx,
                       const uint32_t &m_offset) const
```
Another small method, that just checks for grouped GEMM cases whether the current block of rows that will be computed in the WGMMA start with an expert index or a padding index. If it starts with a padding index, then the entire tile is invalid and there is no point in wasting SM resources on it.

That's all for the DeepGEMM Tile scheduler. I'm going to implement this and integrate it into my current FP8 GEMM kernel. Another scheduler to explore is the more powerful, expressive [Cutlass Schedulers](https://github.com/NVIDIA/cutlass/blob/main/include/cutlass/gemm/kernel/sm90_tile_scheduler_group.hpp#L130) and how it implements Group and Stream-k scheduling with heuristics.
