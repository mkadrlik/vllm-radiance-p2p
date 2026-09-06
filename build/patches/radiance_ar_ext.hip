// radiance_ar_ext.hip: multi-block one-shot P2P-BAR all-reduce for dual R9700
// (gfx1201/RDNA4, TP=2, PCIe). Pure pybind11 + HIP (wheel-ROCm ships no rocThrust,
// which torch/extension.h needs). Python passes raw data_ptr / numel / dtype / stream.
//
// Design:
//  * PUSH model: each rank writes ITS input into the PEER's IPC scratch, then reduces
//    (local input + peer's data now in MY scratch) -> out. Only scratch+flags are
//    IPC-shared; input/output stay local -> no cudagraph buffer registration needed.
//  * MULTI-BLOCK, per-chunk independent flags (NO grid barrier): block b owns a
//    contiguous chunk of 16B words -> pushes it to peer, sets peer_flags[b], spins on
//    my_flags[b], reduces its chunk. Spreads the push across CUs (PCIe saturation) and
//    the reduce across CUs (HBM saturation): the single-block bandwidth fix.
//  * cudagraph-safe: seq is a DEVICE-resident PER-BLOCK counter the kernel increments.
//    Both ranks execute the identical graph sequence (SPMD) so seq_ctr[b] stays in
//    lockstep across ranks regardless of which batch-size graphs ran.
//  * race-free reuse: scratch DOUBLE-BUFFERED by (seq&1); same-stream serialization +
//    the per-block flag handshake guarantee a slot is drained before reuse.
//  * FINE-GRAINED memory (hipDeviceMallocFinegrained) => uncached => posted straight to
//    the fabric => SCOPE_SYS release/acquire skip the L2 writeback. drain/acq explicit
//    fences are tunable (all-thread system fences; default drain=1/acq=0 on fine-grained).
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hip/hip_bf16.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>
namespace py = pybind11;

#define RADIANCE_SPIN_MAX 4000000000ULL
#define RADIANCE_MAX_BLOCKS 512
#define HIPOK(e) do { hipError_t _e = (e); if (_e != hipSuccess) \
    throw std::runtime_error(std::string("hip: ") + hipGetErrorString(_e)); } while (0)

__device__ __forceinline__ void store_sys_rel(unsigned int* p, unsigned int v) {
  __hip_atomic_store(p, v, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
}
__device__ __forceinline__ unsigned int load_sys_acq(const unsigned int* p) {
  return __hip_atomic_load(p, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM);
}
__device__ __forceinline__ float rd_to_f(const float& x) { return x; }
__device__ __forceinline__ float rd_to_f(const __half& x) { return __half2float(x); }
__device__ __forceinline__ float rd_to_f(const __hip_bfloat16& x) { return (float)x; }
__device__ __forceinline__ void rd_from_f(float v, float& o) { o = v; }
__device__ __forceinline__ void rd_from_f(float v, __half& o) { o = __float2half(v); }
__device__ __forceinline__ void rd_from_f(float v, __hip_bfloat16& o) { o = (__hip_bfloat16)v; }

// vectorized 128-bit add: an int4 packs LANES elements of T; fp32-accumulate reduce.
template <typename T, int LANES>
__device__ __forceinline__ int4 vadd(const int4& a, const int4& b) {
  int4 r;
  const T* pa = reinterpret_cast<const T*>(&a);
  const T* pb = reinterpret_cast<const T*>(&b);
  T* pr = reinterpret_cast<T*>(&r);
#pragma unroll
  for (int j = 0; j < LANES; ++j) rd_from_f(rd_to_f(pa[j]) + rd_to_f(pb[j]), pr[j]);
  return r;
}

// Multi-block one-shot all-reduce. Each block owns a contiguous chunk of 16B-word
// space and drives its own flag handshake; fully independent, no cross-block sync.
// TWO-PHASE handshake (data flag + done flag), same protocol as vLLM's ca_comm:
//   before pushing into peer slot (s&1), wait my_done[b] >= s-2  — the peer's
//   launch s-2 may STILL be reducing out of that slot; without this wait, a
//   back-to-back replay of the same graph overwrites peer scratch mid-read and
//   corrupts the result (observed in production as first-decode garbage ->
//   divergent MTP draft decisions -> AR count divergence -> mutual spin).
//   after reducing, post done s to the peer so it may reuse the slot.
template <typename T, int LANES>
__global__ void radiance_ar_mb(
    const int4* __restrict__ in4, int4* peer_base, const int4* my_base, int4* __restrict__ out4,
    unsigned int* peer_flags, unsigned int* my_flags,
    unsigned int* peer_done, unsigned int* my_done, unsigned int* seq_ctrs,
    int n16, int slot_stride16, int drain, int acq, unsigned long long spin_max) {
  const int b = blockIdx.x, nb = gridDim.x, tid = threadIdx.x, nt = blockDim.x;
  __shared__ unsigned int s_seq;
  if (tid == 0) s_seq = atomicAdd(&seq_ctrs[b], 1u) + 1u;  // per-block, replay-safe
  __syncthreads();
  const unsigned int s = s_seq;
  const int slot = (int)(s & 1u);
  int4* ps = peer_base + (size_t)slot * slot_stride16;
  const int4* ms = my_base + (size_t)slot * slot_stride16;

  // FIXED word partition: block b owns slot words [b*chunk16, (b+1)*chunk16),
  // chunk16 = ceil(slot16/nb). The Python side guarantees nb is CONSTANT across
  // all launches (fixed grid = max_nb; empty-range blocks still handshake), so a
  // word range is written and read by the SAME block index on both ranks for
  // every graph — that is what makes the per-block done guard sound. The old
  // size-proportional grid+partition let one rank's block b write at launch s
  // overlap the peer's block b'!=b read at s-2 when different-size graphs
  // interleaved — done[b] missed b' and the slot corrupted silently
  // (observed: t3/t4 mismatches with zero SPIN-ABORTs).
  const int chunk16 = (slot_stride16 + nb - 1) / nb;
  const int start = b * chunk16;
  int end = start + chunk16; if (end > n16) end = n16;

  // slot-reuse guard: wait until the peer finished READING this slot at s-2.
  // Signed wrap-safe comparison; at steady state the peer is caught up and this
  // does not spin at all. One thread polls, the block syncs before pushing.
  // Abort log: high bit set = done-wait (slot reuse) timeout, clear = data-wait.
  if (tid == 0 && s >= 3) {
    unsigned long long z = 0;
    while ((int)(load_sys_acq(&my_done[b]) - (s - 2u)) < 0) {
      if (++z > spin_max) { atomicMax(&seq_ctrs[RADIANCE_MAX_BLOCKS + b], 0x80000000u | s); break; }
    }
  }
  __syncthreads();

  // push chunk -> peer slot, 4x-unrolled for memory-level parallelism (PCIe saturation)
  {
    int k = start + tid;
    for (; k + 3 * nt < end; k += 4 * nt) {
      int4 v0 = in4[k], v1 = in4[k + nt], v2 = in4[k + 2 * nt], v3 = in4[k + 3 * nt];
      ps[k] = v0; ps[k + nt] = v1; ps[k + 2 * nt] = v2; ps[k + 3 * nt] = v3;
    }
    for (; k < end; k += nt) ps[k] = in4[k];
  }
  // drain modes (ordering of the pushes vs the release flag, so the peer sees data first):
  //  0 = none (relies purely on PCIe posted-write ordering + the release flag)
  //  1 = all-thread __threadfence_system  (safe but emits an L2 wb-invalidate per
  //      wavefront (WASTED on fine-grained/uncached memory; floods at high nt*nb)
  //  2 = single-thread __threadfence_system (cheap but only drains wavefront 0's stores)
  //  3 = all-thread store-completion wait (s_wait_storecnt 0): every wavefront drains
  //      its posted writes into the fabric with NO L2 wb-invalidate; PCIe keeps posted
  //      writes to the same peer ordered, so the later release flag lands after the data.
  //      Correct for FINE-GRAINED memory (uncached; coarse would leave data in L2).
  if (drain == 1) __threadfence_system();
  else if (drain == 3) {
#if defined(__gfx1100__) || defined(__gfx1101__) || defined(__gfx1102__) || defined(__gfx1103__)
    // RDNA3: export-completion wait drains posted stores (s_wait_storecnt is gfx12+ syntax)
    asm volatile("s_waitcnt expcnt(0)" ::: "memory");
#else
    asm volatile("s_wait_storecnt 0x0" ::: "memory");
#endif
  }
  __syncthreads();
  if (tid == 0) {
    if (drain == 2) __threadfence_system();
    store_sys_rel(&peer_flags[b], s);
    // Flush the flag store ONTO THE WIRE before spinning. Peer-directed stores on
    // RDNA3 ride the export path (same reason drain=3 uses expcnt for the data
    // push): a bare release store can sit in the write-combining buffer while
    // tid0 spins on the peer's mirror-image flag. If BOTH ranks buffer their flag,
    // neither ever sees the other's post until a spin timeout + kernel-exit membar
    // flushes it — the exact production signature (both ranks SPIN-ABORT on the
    // same seq/blocks, flags arrive only after the timeout). vmcnt covers global
    // atomics, expcnt covers export stores: wait for both.
    asm volatile("s_waitcnt vmcnt(0) expcnt(0)" ::: "memory");
    unsigned long long z = 0;
    while (load_sys_acq(&my_flags[b]) < s) {
      if (++z > spin_max) {
        // Timed out: the peer never posted this seq -> the ranks' AR sequences
        // diverged (SPMD lockstep broken). Publish that fact device-side so a
        // host sampler can see WHICH block gave up at WHICH seq instead of the
        // kernel silently producing garbage.
        atomicMax(&seq_ctrs[RADIANCE_MAX_BLOCKS + b], s);
        break;
      }
    }
  }
  __syncthreads();
  if (acq) __threadfence_system();
  // reduce chunk (vectorized, 4x-unrolled): out = my_input + peer_input(in my scratch)
  {
    int k = start + tid;
    for (; k + 3 * nt < end; k += 4 * nt) {
      out4[k]          = vadd<T, LANES>(in4[k],          ms[k]);
      out4[k + nt]     = vadd<T, LANES>(in4[k + nt],     ms[k + nt]);
      out4[k + 2 * nt] = vadd<T, LANES>(in4[k + 2 * nt], ms[k + 2 * nt]);
      out4[k + 3 * nt] = vadd<T, LANES>(in4[k + 3 * nt], ms[k + 3 * nt]);
    }
    for (; k < end; k += nt) out4[k] = vadd<T, LANES>(in4[k], ms[k]);
  }
  // Signal the peer that slot (s&1) is fully READ. __syncthreads() alone is an
  // EXECUTION barrier — the other threads' vector loads may still be IN FLIGHT
  // when it passes (this is what corrupted back-to-back replays: done posted
  // early, peer overwrote the slot mid-read). Every wavefront must waitcnt its
  // own vmcnt first; then tid0's release store + wire-flush publishes done[s].
  asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
  __syncthreads();
  if (tid == 0) {
    store_sys_rel(&peer_done[b], s);
    asm volatile("s_waitcnt vmcnt(0) expcnt(0)" ::: "memory");
  }
}

static std::tuple<int64_t, py::bytes> alloc_shared(int64_t size, bool finegrained) {
  void* p = nullptr;
  if (finegrained) HIPOK(hipExtMallocWithFlags(&p, size, hipDeviceMallocFinegrained));
  else HIPOK(hipMalloc(&p, size));
  HIPOK(hipMemset(p, 0, size));
  hipIpcMemHandle_t h;
  HIPOK(hipIpcGetMemHandle(&h, p));
  return std::make_tuple((int64_t)p, py::bytes((const char*)&h, sizeof(h)));
}
static int64_t open_shared(const std::string& handle) {
  hipIpcMemHandle_t h;
  if (handle.size() != sizeof(h)) throw std::runtime_error("bad ipc handle size");
  std::memcpy(&h, handle.data(), sizeof(h));
  void* p = nullptr;
  HIPOK(hipIpcOpenMemHandle(&p, h, hipIpcMemLazyEnablePeerAccess));
  return (int64_t)p;
}
static void free_shared(int64_t p) { if (p) (void)hipFree((void*)p); }
static void memzero(int64_t p, int64_t size) { if (p) HIPOK(hipMemset((void*)p, 0, size)); }

// Host-side snapshot of a device u32 array on the CALLER's stream (dst must be
// pinned host memory). Async so the sampler thread never blocks the GIL on the
// compute queue: a spin kernel there could run for the full timeout.
static void copy_u32_async(int64_t dst, int64_t src, int64_t count, int64_t stream_i) {
  HIPOK(hipMemcpyAsync((void*)dst, (const void*)src, (size_t)count * 4,
                       hipMemcpyDeviceToHost, (hipStream_t)stream_i));
}

// single-process 2-GPU testing only: enable direct peer access (the runtime path uses IPC handles)
static void enable_peer(int64_t peer) {
  hipError_t e = hipDeviceEnablePeerAccess((int)peer, 0);
  if (e == hipErrorPeerAccessAlreadyEnabled) { (void)hipGetLastError(); return; }
  if (e != hipSuccess)
    throw std::runtime_error(std::string("enable_peer: ") + hipGetErrorString(e));
}

// spin_max: bounded peer-flag wait. 0 = compile-time default. A short value makes a
// diverged handshake abort quickly (and log to the abort half of seq_ctrs) instead of
// spinning ~4000s — used by the repro harness to turn a deadlock into a fast failure.
static void all_reduce_mb(int64_t peer_scratch, int64_t my_scratch, int64_t peer_flags,
                          int64_t my_flags, int64_t peer_done, int64_t my_done,
                          int64_t seq_ctrs, int64_t slot_stride16,
                          int64_t inp, int64_t out, int64_t n_elem, int64_t dtype,
                          int64_t stream_i, int64_t nblocks, int64_t nthreads,
                          int64_t drain, int64_t acq, int64_t spin_max) {
  hipStream_t st = (hipStream_t)stream_i;
  const unsigned long long spin = spin_max > 0 ? (unsigned long long)spin_max
                                               : RADIANCE_SPIN_MAX;
  const int esize = (dtype == 2) ? 4 : 2;
  const long nbytes = (long)n_elem * esize;
  if (nbytes % 16 != 0) throw std::runtime_error("all_reduce: n_elem*esize not 16B-aligned");
  const int n16 = (int)(nbytes / 16);
  int nb = (int)nblocks;
  if (nb < 1) nb = 1;
  // NO clamp to n16: the grid size must be CONSTANT across launches or the fixed
  // word partition + per-block done guard become unsound (see kernel comment).
  // Tiny messages leave most blocks handshake-only, which is correct and cheap.
  if (nb > RADIANCE_MAX_BLOCKS) nb = RADIANCE_MAX_BLOCKS;
  int nt = (int)nthreads;
  if (nt < 64) nt = 64;
  if (nt > 1024) nt = 1024;
#define L(T, LN) radiance_ar_mb<T, LN><<<nb, nt, 0, st>>>(                                 \
    (const int4*)inp, (int4*)peer_scratch, (const int4*)my_scratch, (int4*)out,           \
    (unsigned*)peer_flags, (unsigned*)my_flags, (unsigned*)peer_done, (unsigned*)my_done, \
    (unsigned*)seq_ctrs, n16,                                                             \
    (int)slot_stride16, (int)drain, (int)acq, spin)
  if (dtype == 0) L(__hip_bfloat16, 8);
  else if (dtype == 1) L(__half, 8);
  else if (dtype == 2) L(float, 4);
  else throw std::runtime_error("all_reduce: bad dtype");
#undef L
}

PYBIND11_MODULE(radiance_ar_ext, m) {
  m.def("alloc_shared", &alloc_shared, py::arg("size"), py::arg("finegrained") = true);
  m.def("open_shared", &open_shared);
  m.def("free_shared", &free_shared);
  m.def("memzero", &memzero);
  m.def("copy_u32_async", &copy_u32_async);
  m.def("enable_peer", &enable_peer);
  m.def("all_reduce_mb", &all_reduce_mb, py::arg("peer_scratch"), py::arg("my_scratch"),
        py::arg("peer_flags"), py::arg("my_flags"), py::arg("peer_done"), py::arg("my_done"),
        py::arg("seq_ctrs"), py::arg("slot_stride16"), py::arg("inp"), py::arg("out"),
        py::arg("n_elem"), py::arg("dtype"), py::arg("stream_i"), py::arg("nblocks"),
        py::arg("nthreads"), py::arg("drain"), py::arg("acq"), py::arg("spin_max") = 0);
  m.attr("MAX_BLOCKS") = RADIANCE_MAX_BLOCKS;
}
