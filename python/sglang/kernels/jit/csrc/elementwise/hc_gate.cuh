#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/math.cuh>
#include <sgl_kernel/tile.cuh>
#include <sgl_kernel/type.cuh>
#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/vec.cuh>
#include <sgl_kernel/warp.cuh>

#include <tvm/ffi/container/tensor.h>

namespace sglang {

struct HcGateParams {
  const void* normed_residual;  // [M, HC * H]
  const void* inject_weight;    // [HC, HC * H]
  float* gate;                  // [M, HC]
};

// Exact phase-1 extraction of hc_combine_kernel. Keeping the 256-thread
// mapping, accumulation order and sigmoid expression unchanged makes the gate
// bit-identical to the gate that the fused combine would have used.
template <int64_t kHcCount, int64_t kHiddenSize, bool kUsePDL, typename Float>
__global__ __launch_bounds__(256) void hc_gate_kernel(
    const HcGateParams __grid_constant__ params) {
  using namespace device;
  using Float2 = packed_t<Float>;
  using Storage = AlignedVector<Float2, 4>;
  constexpr uint32_t kVecLen = 8;
  constexpr uint32_t kNumThreads = 256;
  constexpr int64_t kRowSize = kHcCount * kHiddenSize;
  constexpr uint32_t kVecsPerRow = kRowSize / kVecLen;
  constexpr uint32_t kVecsPerThread = kVecsPerRow / kNumThreads;
  constexpr uint32_t kNumWarps = kNumThreads / kWarpThreads;

  const auto gmem = tile::Memory<Storage>::cta(kNumThreads);
  const uint32_t m = blockIdx.x;
  const auto n_ptr =
      pointer::offset<Float>(params.normed_residual, static_cast<int64_t>(m) * kRowSize);
  const auto w_ptr = static_cast<const Float*>(params.inject_weight);

  PDLWaitPrimary<kUsePDL>();

  Storage n_vec[kVecsPerThread];
#pragma unroll
  for (uint32_t j = 0; j < kVecsPerThread; ++j) {
    n_vec[j] = gmem.load(n_ptr, j);
  }

  float acc[kHcCount];
#pragma unroll
  for (int c = 0; c < kHcCount; ++c) {
    const auto wc_ptr = w_ptr + static_cast<int64_t>(c) * kRowSize;
    float sum = 0.0f;
#pragma unroll
    for (uint32_t j = 0; j < kVecsPerThread; ++j) {
      const Storage w_vec = gmem.load(wc_ptr, j);
#pragma unroll
      for (uint32_t i = 0; i < kVecLen / 2; ++i) {
        const auto [nx, ny] = cast<fp32x2_t>(n_vec[j][i]);
        const auto [wx, wy] = cast<fp32x2_t>(w_vec[i]);
        sum += nx * wx + ny * wy;
      }
    }
    acc[c] = warp::reduce_sum(sum);
  }

  __shared__ float partial[kHcCount][kNumWarps];
  const uint32_t warp_id = threadIdx.x / kWarpThreads;
  const uint32_t lane = threadIdx.x % kWarpThreads;
  if (lane == 0) {
#pragma unroll
    for (int c = 0; c < kHcCount; ++c) {
      partial[c][warp_id] = acc[c];
    }
  }
  __syncthreads();

  if (threadIdx.x < kHcCount) {
    float total = 0.0f;
#pragma unroll
    for (uint32_t w = 0; w < kNumWarps; ++w) {
      total += partial[threadIdx.x][w];
    }
    params.gate[static_cast<int64_t>(m) * kHcCount + threadIdx.x] =
        2.0f / (1.0f + math::exp(-total / kHcCount));
  }

  PDLTriggerSecondary<kUsePDL>();
}

template <int64_t kHcCount, int64_t kHiddenSize, bool kUsePDL, typename DType>
struct HcGateKernel {
  static_assert(sizeof(DType) == 2, "HcGate only supports 2-byte dtypes");
  static_assert(kHcCount > 0);
  static_assert(kHiddenSize > 0 && kHiddenSize % 8 == 0);
  static_assert((kHcCount * kHiddenSize) % (256 * 8) == 0);
  static constexpr auto kernel = hc_gate_kernel<kHcCount, kHiddenSize, kUsePDL, DType>;

  static void run(const tvm::ffi::TensorView normed_residual,
                  const tvm::ffi::TensorView inject_weight,
                  const tvm::ffi::TensorView gate) {
    using namespace host;
    auto M = SymbolicSize{"num_tokens"};
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();

    TensorMatcher({M, kHcCount * kHiddenSize})
        .with_dtype<DType>()
        .with_device(device)
        .verify(normed_residual);
    TensorMatcher({kHcCount, kHcCount * kHiddenSize})
        .with_dtype<DType>()
        .with_device(device)
        .verify(inject_weight);
    TensorMatcher({M, kHcCount})
        .with_dtype<fp32_t>()
        .with_device(device)
        .verify(gate);

    const auto params = HcGateParams{
        .normed_residual = normed_residual.data_ptr(),
        .inject_weight = inject_weight.data_ptr(),
        .gate = static_cast<float*>(gate.data_ptr()),
    };
    LaunchKernel(static_cast<uint32_t>(M.unwrap()), 256, device.unwrap())
        .enable_pdl(kUsePDL)(kernel, params);
  }
};

}  // namespace sglang
