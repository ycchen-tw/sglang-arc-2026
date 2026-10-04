#include "flashinfer/comm/trtllm_mnnvl_allreduce.cuh"
#include "tvm_ffi_utils.h"

// Qwen4-only prototype: preserve FlashInfer's Lamport protocol and reduction
// order, but replace the final [M, H] store with the HC apply epilogue
//
//   out[m, c, h] = residual[m, c, h] + gate[m, c] * allreduce[m, h]
//
// gate is precomputed FP32 and already includes 2 * sigmoid(dot / hc_count).
// The existing FlashInfer op/ABI is intentionally untouched: this source builds
// under a distinct JIT module name and reuses an existing MNNVL workspace.

using namespace flashinfer::trtllm_mnnvl_allreduce;

namespace qwen4_mnnvl_hc {

constexpr int kHcCount = 4;

template <typename T>
struct HcKernelParams {
  AllReduceKernelParams<T> ar;
  T const* residual;
  float const* gate;
  T* output;
};

template <typename T, typename PackedType = float4>
__device__ __forceinline__ void apply_hc_epilogue(
    HcKernelParams<T> const& params, int token, int token_offset,
    PackedVec<PackedType, T> const& reduced) {
  constexpr int kElts = sizeof(PackedType) / sizeof(T);
  int const hidden = params.ar.tokenDim;
#pragma unroll
  for (int c = 0; c < kHcCount; ++c) {
    int64_t const output_offset =
        (static_cast<int64_t>(token) * kHcCount + c) * hidden + token_offset;
    PackedVec<PackedType, T> residual;
    residual.packed = loadPacked<PackedType>(&params.residual[output_offset]);
    PackedVec<PackedType, T> result;
    float const a = params.gate[token * kHcCount + c];
#pragma unroll
    for (int i = 0; i < kElts; ++i) {
      // Match hc_combine.cuh: conversion to FP32, r + a * y, then one cast.
      result.elements[i] = utils::fromFloat<T>(
          utils::toFloat<T>(residual.elements[i]) +
          a * utils::toFloat<T>(reduced.elements[i]));
    }
    reinterpret_cast<PackedType*>(&params.output[output_offset])[0] = result.packed;
  }
}

template <uint8_t WorldSize, typename T, typename PackedType = float4>
__global__ void __launch_bounds__(1024)
    oneshot_hc_kernel(HcKernelParams<T> params) {
  constexpr int kElts = sizeof(PackedType) / sizeof(T);
  constexpr uint32_t kEltSize = sizeof(T);
  namespace cg = cooperative_groups;
  cg::cluster_group cluster = cg::this_cluster();
  int const packed_idx = cluster.thread_rank();
  int const token = blockIdx.x;
  int const token_offset = packed_idx * kElts;
  int const thread_offset = token * params.ar.tokenDim + token_offset;

  cudaGridDependencySynchronize();
  constexpr bool kUseLamportCga = true;
  LamportFlags<PackedType, kUseLamportCga> flag(params.ar.bufferFlags, 1);
  T* stage_mcast = reinterpret_cast<T*>(flag.getCurLamportBuf(params.ar.mcastPtr, 0));
  T* stage_local = reinterpret_cast<T*>(
      flag.getCurLamportBuf(params.ar.inputPtrs[params.ar.rank], 0));

  if (token_offset >= params.ar.tokenDim) {
    flag.ctaArrive();
    flag.clearDirtyLamportBuf(params.ar.inputPtrs[params.ar.rank], -1);
    return;
  }

  PackedVec<PackedType, T> val;
  val.packed = loadPacked<PackedType>(&params.ar.shardPtr[thread_offset]);
  sanitizeLamportPayload<PackedType, T>(val);
  reinterpret_cast<PackedType*>(
      &stage_mcast[token * params.ar.tokenDim * WorldSize +
                   params.ar.rank * params.ar.tokenDim])[packed_idx] = val.packed;

  flag.ctaArrive();
  flag.clearDirtyLamportBuf(params.ar.inputPtrs[params.ar.rank], -1);

  PackedVec<PackedType, T> reduced;
  if constexpr (WorldSize <= 8) {
    reduced = val;
#define QWEN4_RUN_LOCAL_RANK(LOCAL_RANK)                                                \
  case LOCAL_RANK:                                                                      \
    if constexpr (WorldSize > LOCAL_RANK) {                                             \
      PackedVec<PackedType, T> remote_values[WorldSize];                                \
      utils::waitOneshotRemoteRanks<WorldSize, LOCAL_RANK, T, PackedType, kElts>(       \
          remote_values, stage_local, token, params.ar.tokenDim, packed_idx);           \
      reduced = utils::reduceOneshotDeterministic<WorldSize, LOCAL_RANK, T, PackedType, \
                                                   kElts>(remote_values, val);           \
    }                                                                                   \
    break
    switch (params.ar.rank) {
      QWEN4_RUN_LOCAL_RANK(0);
      QWEN4_RUN_LOCAL_RANK(1);
      QWEN4_RUN_LOCAL_RANK(2);
      QWEN4_RUN_LOCAL_RANK(3);
      QWEN4_RUN_LOCAL_RANK(4);
      QWEN4_RUN_LOCAL_RANK(5);
      QWEN4_RUN_LOCAL_RANK(6);
      QWEN4_RUN_LOCAL_RANK(7);
    }
#undef QWEN4_RUN_LOCAL_RANK
  } else {
    reduced = utils::reduceLamportRanksChunked<WorldSize, 8, T, PackedType, kElts>(
        stage_local, token, params.ar.tokenDim, token_offset);
  }

  cudaTriggerProgrammaticLaunchCompletion();
  apply_hc_epilogue<T, PackedType>(params, token, token_offset, reduced);
  flag.waitAndUpdate({static_cast<uint32_t>(params.ar.numTokens * params.ar.tokenDim *
                                            WorldSize * kEltSize),
                      0, 0, 0});
}

template <uint8_t WorldSize, typename T, typename PackedType = float4>
__global__ __launch_bounds__(128) void twoshot_hc_kernel(HcKernelParams<T> params) {
  constexpr int kElts = sizeof(PackedType) / sizeof(T);
  constexpr uint32_t kEltSize = sizeof(T);
  int const packed_idx = blockIdx.y * blockDim.x + threadIdx.x;
  int const token = blockIdx.x;
  int const token_offset = packed_idx * kElts;
  int const thread_offset = token * params.ar.tokenDim + token_offset;
  bool const in_bounds = token_offset < params.ar.tokenDim;
  int const dest_rank = token % WorldSize;
  int const dest_token_offset = token / WorldSize;

  cudaGridDependencySynchronize();
  LamportFlags<PackedType> flag(params.ar.bufferFlags, MNNVLTwoShotStage::NUM_STAGES);
  T* scatter_local = reinterpret_cast<T*>(
      flag.getCurLamportBuf(params.ar.inputPtrs[params.ar.rank], MNNVLTwoShotStage::SCATTER));
  T* scatter_dest = reinterpret_cast<T*>(
      flag.getCurLamportBuf(params.ar.inputPtrs[dest_rank], MNNVLTwoShotStage::SCATTER));
  T* broadcast_write = reinterpret_cast<T*>(
      flag.getCurLamportBuf(params.ar.mcastPtr, MNNVLTwoShotStage::BROADCAST));
  T* broadcast_read = reinterpret_cast<T*>(
      flag.getCurLamportBuf(params.ar.inputPtrs[params.ar.rank], MNNVLTwoShotStage::BROADCAST));

  PackedVec<PackedType, T> val;
  if (in_bounds) {
    val.packed = loadPacked<PackedType>(&params.ar.shardPtr[thread_offset]);
    sanitizeLamportPayload<PackedType, T>(val);
    reinterpret_cast<PackedType*>(
        &scatter_dest[dest_token_offset * params.ar.tokenDim * WorldSize +
                      params.ar.rank * params.ar.tokenDim])[packed_idx] = val.packed;
  }

  cudaTriggerProgrammaticLaunchCompletion();
  flag.clearDirtyLamportBuf(params.ar.inputPtrs[params.ar.rank], MNNVLTwoShotStage::SCATTER);

  if (in_bounds && dest_rank == params.ar.rank) {
    int const local_token = token / WorldSize;
    constexpr int kRankChunk = WorldSize < 16 ? WorldSize : 16;
    PackedVec<PackedType, T> reduced =
        utils::reduceLamportRanksChunked<WorldSize, kRankChunk, T, PackedType, kElts>(
            scatter_local, local_token, params.ar.tokenDim, token_offset);
    sanitizeLamportPayload<PackedType, T>(reduced);
    reinterpret_cast<PackedType*>(
        &broadcast_write[token * params.ar.tokenDim])[packed_idx] = reduced.packed;
  }

  flag.clearDirtyLamportBuf(params.ar.inputPtrs[params.ar.rank], MNNVLTwoShotStage::BROADCAST);
  flag.ctaArrive();
  if (in_bounds) {
    auto loaded = loadPackedVolatile<PackedType>(&broadcast_read[thread_offset]);
    while (isLamportDirty(loaded)) {
      loaded = loadPackedVolatile<PackedType>(&broadcast_read[thread_offset]);
    }
    PackedVec<PackedType, T> reduced;
    reduced.packed = loaded.packed;
    apply_hc_epilogue<T, PackedType>(params, token, token_offset, reduced);
  }

  flag.waitAndUpdate(
      {static_cast<uint32_t>(((params.ar.numTokens + WorldSize - 1) / WorldSize * WorldSize) *
                             params.ar.tokenDim * kEltSize),
       static_cast<uint32_t>(params.ar.numTokens * params.ar.tokenDim * kEltSize), 0, 0});
}

template <typename T, typename PackedType = float4>
__global__ __launch_bounds__(128) void standalone_hc_epilogue_kernel(
    T const* reduced, T const* residual, float const* gate, T* output,
    int num_tokens, int hidden) {
  constexpr int kElts = sizeof(PackedType) / sizeof(T);
  int const packed_idx = blockIdx.y * blockDim.x + threadIdx.x;
  int const token_offset = packed_idx * kElts;
  if (token_offset >= hidden) return;
  int const token = blockIdx.x;
  PackedVec<PackedType, T> value;
  value.packed = loadPacked<PackedType>(&reduced[token * hidden + token_offset]);
  AllReduceKernelParams<T> ar{};
  ar.tokenDim = hidden;
  HcKernelParams<T> params{ar, residual, gate, output};
  apply_hc_epilogue<T, PackedType>(params, token, token_offset, value);
}

template <typename T>
HcKernelParams<T> make_kernel_params(
    TensorView input, int64_t multicast_buffer_ptr, int64_t buffer_ptrs_dev,
    int64_t buffer_ptr_local, TensorView buffer_flags, int nranks, int rank,
    TensorView residual, TensorView gate, TensorView output) {
  AllReduceKernelParams<T> ar{
      .outputPtr = nullptr,
      .prenormedPtr = nullptr,
      .shardPtr = reinterpret_cast<T const*>(input.data_ptr()),
      .residualInPtr = nullptr,
      .gammaPtr = nullptr,
      .inputPtrs = reinterpret_cast<T**>(buffer_ptrs_dev),
      .mcastPtr = reinterpret_cast<T*>(multicast_buffer_ptr),
      .localBufferPtr = reinterpret_cast<T*>(buffer_ptr_local),
      .nRanks = nranks,
      .rank = rank,
      .numTokens = static_cast<int>(input.size(0)),
      .tokenDim = static_cast<int>(input.size(1)),
      .epsilon = 0.f,
      .weightBias = 0.f,
      .outputScalePtr = nullptr,
      .quantOutPtr = nullptr,
      .scalingFactorOutPtr = nullptr,
      .sfLayout = flashinfer::QuantizationSFLayout::SWIZZLED_128x4,
      .bufferFlags = reinterpret_cast<uint32_t*>(buffer_flags.data_ptr()),
  };
  return HcKernelParams<T>{
      ar,
      reinterpret_cast<T const*>(residual.data_ptr()),
      reinterpret_cast<float const*>(gate.data_ptr()),
      reinterpret_cast<T*>(output.data_ptr()),
  };
}

template <typename T>
cudaError_t dispatch(HcKernelParams<T> const& params, bool launch_with_pdl,
                     bool use_oneshot, cudaStream_t stream) {
  if (use_oneshot) {
    int const elts_per_thread = sizeof(float4) / sizeof(T);
    auto [block_size, cluster_size, loads_per_thread] =
        utils::adjustGridConfig(params.ar.numTokens, params.ar.tokenDim,
                                elts_per_thread, true);
    if (block_size > 1024 || loads_per_thread != 1 ||
        cluster_size > utils::kMaxClusterSize) {
      return cudaErrorInvalidConfiguration;
    }
    cudaLaunchAttribute attrs[2];
    attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attrs[0].val.programmaticStreamSerializationAllowed = launch_with_pdl ? 1 : 0;
    attrs[1].id = cudaLaunchAttributeClusterDimension;
    attrs[1].val.clusterDim.x = 1;
    attrs[1].val.clusterDim.y = cluster_size;
    attrs[1].val.clusterDim.z = 1;
    cudaLaunchConfig_t config{
        .gridDim = dim3(params.ar.numTokens, cluster_size, 1),
        .blockDim = static_cast<uint32_t>(block_size),
        .dynamicSmemBytes = 0,
        .stream = stream,
        .attrs = attrs,
        .numAttrs = 2,
    };
#define QWEN4_LAUNCH_ONESHOT(WORLD_SIZE) \
  FLASHINFER_CUDA_CALL(cudaLaunchKernelEx( \
      &config, &oneshot_hc_kernel<WORLD_SIZE, T>, params))
    switch (params.ar.nRanks) {
      case 2: QWEN4_LAUNCH_ONESHOT(2); break;
      case 4: QWEN4_LAUNCH_ONESHOT(4); break;
      case 8: QWEN4_LAUNCH_ONESHOT(8); break;
      default: return cudaErrorInvalidValue;
    }
#undef QWEN4_LAUNCH_ONESHOT
    return cudaSuccess;
  }

  int const packed = sizeof(float4) / sizeof(T);
  int const packed_cols = (params.ar.tokenDim + packed - 1) / packed;
  int const blocks_per_token = (packed_cols + 127) / 128;
  cudaLaunchAttribute attrs[1];
  attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attrs[0].val.programmaticStreamSerializationAllowed = launch_with_pdl ? 1 : 0;
  cudaLaunchConfig_t config{
      .gridDim = dim3(params.ar.numTokens, blocks_per_token, 1),
      .blockDim = 128,
      .dynamicSmemBytes = 0,
      .stream = stream,
      .attrs = attrs,
      .numAttrs = 1,
  };
#define QWEN4_LAUNCH_TWOSHOT(WORLD_SIZE) \
  FLASHINFER_CUDA_CALL(cudaLaunchKernelEx( \
      &config, &twoshot_hc_kernel<WORLD_SIZE, T>, params))
  switch (params.ar.nRanks) {
    case 2: QWEN4_LAUNCH_TWOSHOT(2); break;
    case 4: QWEN4_LAUNCH_TWOSHOT(4); break;
    case 8: QWEN4_LAUNCH_TWOSHOT(8); break;
    default: return cudaErrorInvalidValue;
  }
#undef QWEN4_LAUNCH_TWOSHOT
  return cudaSuccess;
}

void qwen4_mnnvl_hc_combine(
    TensorView input, int64_t multicast_buffer_ptr, int64_t buffer_ptrs_dev,
    int64_t buffer_ptr_local, TensorView buffer_flags, int64_t nranks,
    int64_t rank, bool launch_with_pdl, bool use_oneshot, TensorView residual,
    TensorView gate, TensorView output) {
  ffi::CUDADeviceGuard guard(input.device().device_id);
  auto stream = get_stream(input.device());
  TVM_FFI_ICHECK(input.ndim() == 2 && residual.ndim() == 2 &&
                 gate.ndim() == 2 && output.ndim() == 2);
  TVM_FFI_ICHECK(residual.size(0) == input.size(0) &&
                 residual.size(1) == kHcCount * input.size(1));
  TVM_FFI_ICHECK(output.size(0) == residual.size(0) &&
                 output.size(1) == residual.size(1));
  TVM_FFI_ICHECK(gate.size(0) == input.size(0) && gate.size(1) == kHcCount);
  TVM_FFI_ICHECK(input.dtype() == residual.dtype() && input.dtype() == output.dtype());
  TVM_FFI_ICHECK(encode_dlpack_dtype(gate.dtype()) == float32_code);
  TVM_FFI_ICHECK(nranks == 2 || nranks == 4 || nranks == 8);
  TVM_FFI_ICHECK(rank >= 0 && rank < nranks);

  auto launch = [&]<typename T>() {
    auto params = make_kernel_params<T>(
        input, multicast_buffer_ptr, buffer_ptrs_dev, buffer_ptr_local,
        buffer_flags, static_cast<int>(nranks), static_cast<int>(rank),
        residual, gate, output);
    cudaError_t status = dispatch<T>(params, launch_with_pdl, use_oneshot, stream);
    TVM_FFI_ICHECK(status == cudaSuccess)
        << "qwen4_mnnvl_hc_combine failed: " << cudaGetErrorString(status);
  };
  switch (encode_dlpack_dtype(input.dtype())) {
    case float16_code: launch.template operator()<half>(); break;
    case bfloat16_code: launch.template operator()<__nv_bfloat16>(); break;
    default:
      TVM_FFI_LOG_AND_THROW(NotImplementedError)
          << "qwen4_mnnvl_hc_combine supports only FP16/BF16";
  }
}

void qwen4_hc_epilogue(TensorView reduced, TensorView residual,
                       TensorView gate, TensorView output) {
  ffi::CUDADeviceGuard guard(reduced.device().device_id);
  auto stream = get_stream(reduced.device());
  TVM_FFI_ICHECK(reduced.ndim() == 2 && residual.ndim() == 2 &&
                 gate.ndim() == 2 && output.ndim() == 2);
  int const m = reduced.size(0);
  int const h = reduced.size(1);
  TVM_FFI_ICHECK(residual.size(0) == m && residual.size(1) == kHcCount * h);
  TVM_FFI_ICHECK(output.size(0) == m && output.size(1) == kHcCount * h);
  TVM_FFI_ICHECK(gate.size(0) == m && gate.size(1) == kHcCount);
  TVM_FFI_ICHECK(reduced.dtype() == residual.dtype() &&
                 reduced.dtype() == output.dtype());
  TVM_FFI_ICHECK(encode_dlpack_dtype(gate.dtype()) == float32_code);
  int const packed = sizeof(float4) / 2;
  int const packed_cols = (h + packed - 1) / packed;
  dim3 grid(m, (packed_cols + 127) / 128, 1);
  auto launch = [&]<typename T>() {
    standalone_hc_epilogue_kernel<T><<<grid, 128, 0, stream>>>(
        reinterpret_cast<T const*>(reduced.data_ptr()),
        reinterpret_cast<T const*>(residual.data_ptr()),
        reinterpret_cast<float const*>(gate.data_ptr()),
        reinterpret_cast<T*>(output.data_ptr()), m, h);
  };
  switch (encode_dlpack_dtype(reduced.dtype())) {
    case float16_code: launch.template operator()<half>(); break;
    case bfloat16_code: launch.template operator()<__nv_bfloat16>(); break;
    default:
      TVM_FFI_LOG_AND_THROW(NotImplementedError)
          << "qwen4_hc_epilogue supports only FP16/BF16";
  }
}

}  // namespace qwen4_mnnvl_hc

TVM_FFI_DLL_EXPORT_TYPED_FUNC(qwen4_mnnvl_hc_combine,
                              qwen4_mnnvl_hc::qwen4_mnnvl_hc_combine);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(qwen4_hc_epilogue,
                              qwen4_mnnvl_hc::qwen4_hc_epilogue);
