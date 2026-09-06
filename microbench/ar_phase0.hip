// ar_phase0.hip — Phase 0 microbench: peer-write flush primitives on gfx1100 PCIe.
//
// Measures, on real gfx1100<->gfx1100 fine-grained IPC BAR mappings:
//  (a) store visibility latency per cache-policy variant (ping-pong RTT),
//  (b) data-before-flag ordering: does flush variant X make a payload observable
//      with bounded latency before its flag (the V1 claim under test),
//  (c) packed {payload,seq} single 8B store: tearing + RTT distribution,
//  (d) contention: RTT under concurrent in-kernel read-storm blocks, backoff levels.
//
// gfx1100 ISA reality (probe2.hip, verified): s_timer/s_timerlo/s_timerhi,
// s_memrealtime, clock64() all FAIL assembly on gfx1100 when actually referenced
// from a kernel (clang -O0 probing "passes" only because dead functions are
// dropped before the assembler). There is NO shader-readable wall clock on RDNA3.
// So: the device "clock" is the spin-loop iteration count (one uncached PCIe read
// per poll ≈ 1.3 us, same unit the production kernel's RADIANCE_SPIN_MAX uses),
// and mean wall-clock per handshake comes from host timing of the kernel launch.
//
// SAFETY: every spin is bounded by a counter cap; responder caps are 8x tighter and
// resync by skipping ahead, so a dead initiator never cascades. No vLLM involved.
#include <hip/hip_runtime.h>
#include <pybind11/pybind11.h>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
namespace py = pybind11;

#define HIPOK(e) do { hipError_t _e = (e); if (_e != hipSuccess) \
    throw std::runtime_error(std::string("hip: ") + hipGetErrorString(_e)); } while (0)

// store variants
//  0 plain st.global (default policy)
//  1 __hip_atomic_store RELEASE SCOPE_SYS (what V0 uses)
//  2 __threadfence_system() then plain store
//  3 buffer_store, cache policy 0 (sc0/cached)
//  4 buffer_store glsc (sc1: L2 store, coherent)
//  5 buffer_store sc1 + dependent read-back flush (V3 mechanism; gfx1100 has no wbl2)
//  6 plain store + s_waitcnt vmcnt(0) (V1 flush claim)
//  7 plain store + s_waitcnt vmcnt(0),excnt(0) (V1 drain=3)
enum { SV_PLAIN=0, SV_RELSYS=1, SV_TFS=2, SV_BUF_SC0=3, SV_BUF_SC1=4,
       SV_SC1_READFLUSH=5, SV_PLAIN_VMCNT=6, SV_PLAIN_VMEX=7, SV_PACKED=8, SV_COUNT=9 };

#define BUF_GLSC 0x02000000u  // GLSC cache-policy bit (sc1) for raw_buffer builtins

template <unsigned AUX>
__device__ __forceinline__ void buf_store32(unsigned* p, unsigned v) {
  auto rsrc = __builtin_amdgcn_make_buffer_rsrc((void*)p, 0, 0xFFFFFFFFu, 0);
  __builtin_amdgcn_raw_buffer_store_b32(v, rsrc, 0, 0, AUX);
}
template <unsigned AUX>
__device__ __forceinline__ unsigned buf_load32(const unsigned* p) {
  auto rsrc = __builtin_amdgcn_make_buffer_rsrc((void*)p, 0, 0xFFFFFFFFu, 0);
  return __builtin_amdgcn_raw_buffer_load_b32(rsrc, 0, 0, AUX);
}

__device__ __forceinline__ void store_var(unsigned* p, unsigned v, int variant) {
  switch (variant) {
    case SV_PLAIN: *(volatile unsigned*)p = v; break;
    case SV_RELSYS: __hip_atomic_store(p, v, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM); break;
    case SV_TFS: __threadfence_system(); *(volatile unsigned*)p = v; break;
    case SV_BUF_SC0:
      buf_store32<0>(p, v);
      asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
      break;
    case SV_BUF_SC1:
      buf_store32<BUF_GLSC>(p, v);
      asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
      break;
    case SV_SC1_READFLUSH:
      buf_store32<BUF_GLSC>(p, v);
      asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
      (void)buf_load32<BUF_GLSC>(p);  // dependent non-posted read: forces write completion
      asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
      break;
    case SV_PLAIN_VMCNT:
      *(volatile unsigned*)p = v;
      asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
      break;
    case SV_PLAIN_VMEX:
      *(volatile unsigned*)p = v;
      asm volatile("s_waitcnt vmcnt(0) expcnt(0)" ::: "memory");
      break;
    default: *(volatile unsigned*)p = v; break;
  }
}

// packed 8-byte store: {payload:32, seq:32} in one aligned u64 + store wait.
__device__ __forceinline__ void store_packed(unsigned long long* p, unsigned payload, unsigned seq) {
  *(volatile unsigned long long*)p = ((unsigned long long)seq << 32) | payload;
  asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
}

__device__ __forceinline__ unsigned load_acq(const unsigned* p) {
  return __hip_atomic_load(p, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM);
}
__device__ __forceinline__ unsigned long long load_acq64(const unsigned long long* p) {
  return __hip_atomic_load(p, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM);
}

// backoff between spin polls: nops * 63-cycle s_nop, 0 = hot spin.
__device__ __forceinline__ void backoff(int nops) {
  for (int i = 0; i < nops; ++i) asm volatile("s_nop 63" ::: "memory");
}

// Hammer: extra workgroups of the SAME kernel storm the peer's scratch with uncached
// reads (GPU_MAX_HW_QUEUES=1 serializes streams; same-kernel workgroups stay concurrent).
__device__ __forceinline__ void hammer_block(const unsigned* w, int hammer_iters, int nops) {
  for (int i = threadIdx.x; i < hammer_iters; i += blockDim.x) {
    (void)load_acq(w); (void)load_acq(w + 1); (void)load_acq(w + 2); (void)load_acq(w + 3);
    backoff(nops);
  }
}

// ---- (a) ping-pong RTT. role 0 = initiator (records spin counts), 1 = responder. ----
// u32 slots: [0]=req, [2]=rsp. samples[i] = spin polls until rsp>=i (or ~0ull on abort).
__global__ void pingpong(unsigned* peer_base, unsigned* my_base,
                         unsigned long long* samples, unsigned long long* rsp_spins,
                         unsigned* aborts,
                         int role, int iters, int variant, int nops,
                         unsigned long long cap, int hammer_iters) {
  if (blockIdx.x != 0) {
    if (hammer_iters > 0) hammer_block(peer_base, hammer_iters, nops);
    return;
  }
  if (role == 0) {
    for (int i = 1; i <= iters; ++i) {
      store_var(peer_base + 0, (unsigned)i, variant);
      unsigned long long spins = 0;
      bool done = false;
      while (!done) {
        if (load_acq(my_base + 2) >= (unsigned)i) done = true;
        else if (++spins > cap) { atomicOr(aborts, 1u); break; }
        else backoff(nops);
      }
      samples[i - 1] = done ? spins : ~0ull;
      if (!done) break;
    }
  } else {
    const unsigned long long rcap = cap / 8;  // tighter: never outlive a dead initiator
    int miss = 0;
    for (int i = 1; i <= iters; ++i) {
      unsigned long long spins = 0;
      bool ok = false;
      while (!ok) {
        if (load_acq(my_base + 0) >= (unsigned)i) ok = true;
        else if (++spins > rcap) break;
        else backoff(nops);
      }
      if (!ok) {
        atomicOr(aborts, 2u);
        if (++miss > 3) break;
        continue;
      }
      miss = 0;
      rsp_spins[i - 1] = spins;  // one-way visibility: polls until initiator's store seen
      store_var(peer_base + 2, (unsigned)i, variant);
    }
  }
}

// ---- (b) ordering: data posted with dv then flag with fv; reader checks payload
// on flag arrival. stale>0 => flag outran data (flush failed). Echo backpressure
// keeps RTT honest. u32 slots: [0]=data, [2]=flag, [4]=echo.
__global__ void order_test(unsigned* peer_base, unsigned* my_base,
                           unsigned long long* arrival, unsigned long long* rsp_spins,
                           unsigned* stale,
                           unsigned* flags_never, int role, int iters,
                           int data_variant, int flag_variant, int nops,
                           unsigned long long cap, int hammer_iters) {
  if (blockIdx.x != 0) {
    if (hammer_iters > 0) hammer_block(peer_base, hammer_iters, nops);
    return;
  }
  if (role == 0) {
    for (int i = 1; i <= iters; ++i) {
      store_var(peer_base + 0, 0xA5000000u | (unsigned)i, data_variant);
      store_var(peer_base + 2, (unsigned)i, flag_variant);
      unsigned long long spins = 0;
      bool done = false;
      while (!done) {
        if (load_acq(my_base + 4) >= (unsigned)i) done = true;
        else if (++spins > cap) { atomicOr(flags_never, 1u); break; }
        else backoff(nops);
      }
      arrival[i - 1] = done ? spins : ~0ull;
      if (!done) break;
    }
  } else {
    const unsigned long long rcap = cap / 8;
    int miss = 0;
    for (int i = 1; i <= iters; ++i) {
      unsigned long long spins = 0;
      bool ok = false;
      while (!ok) {
        if (load_acq(my_base + 2) >= (unsigned)i) ok = true;
        else if (++spins > rcap) break;
        else backoff(nops);
      }
      if (!ok) {
        atomicOr(flags_never, 2u);
        if (++miss > 3) break;
        continue;
      }
      miss = 0;
      unsigned d = load_acq(my_base + 0);
      if (d != (0xA5000000u | (unsigned)i)) atomicAdd(stale, 1u);
      rsp_spins[i - 1] = spins | ((d == (0xA5000000u | (unsigned)i) ? 1ull : 0ull) << 63);
      store_var(peer_base + 4, (unsigned)i, flag_variant);
    }
  }
}

// ---- (c) LL packed 8B: u64 slots [2]=mbox {payload,seq}, [3]=echo {payload,seq}. ----
__global__ void ll_test(unsigned long long* peer_base, unsigned long long* my_base,
                        unsigned long long* samples, unsigned long long* rsp_spins,
                        unsigned* tears, unsigned* aborts,
                        int role, int iters, int nops, unsigned long long cap,
                        int hammer_iters) {
  if (blockIdx.x != 0) {
    if (hammer_iters > 0) hammer_block((const unsigned*)peer_base, hammer_iters, nops);
    return;
  }
  const unsigned PAY = 0xBEEF0000u;
  if (role == 0) {
    for (int i = 1; i <= iters; ++i) {
      store_packed(peer_base + 2, PAY ^ (unsigned)i, (unsigned)i);
      unsigned long long spins = 0;
      bool done = false;
      unsigned long long v = 0;
      while (!done) {
        v = load_acq64(my_base + 3);
        if ((v >> 32) >= (unsigned)i) done = true;
        else if (++spins > cap) { atomicOr(aborts, 1u); break; }
        else backoff(nops);
      }
      if (!done) { samples[i - 1] = ~0ull; break; }
      // echo carries the payload the responder read from OUR mbox: mismatch =>
      // the 8B store arrived torn/reordered
      if ((unsigned)v != (PAY ^ (unsigned)i)) atomicAdd(tears, 1u);
      samples[i - 1] = spins;
    }
  } else {
    const unsigned long long rcap = cap / 8;
    int miss = 0;
    for (int i = 1; i <= iters; ++i) {
      unsigned long long spins = 0;
      bool ok = false;
      while (!ok) {
        if ((load_acq64(my_base + 2) >> 32) >= (unsigned)i) ok = true;
        else if (++spins > rcap) break;
        else backoff(nops);
      }
      if (!ok) {
        atomicOr(aborts, 2u);
        if (++miss > 3) break;
        continue;
      }
      miss = 0;
      unsigned long long v = load_acq64(my_base + 2);
      rsp_spins[i - 1] = spins;
      store_packed(peer_base + 3, (unsigned)v, (unsigned)i);
    }
  }
}

// ---- host bindings ----
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
static void memzero(int64_t p, int64_t size) { HIPOK(hipMemset((void*)p, 0, size)); }

static void run_pingpong(int64_t peer, int64_t me, int64_t samples, int64_t rsp_spins,
                         int64_t aborts,
                         int64_t role, int64_t iters, int64_t variant, int64_t nops,
                         int64_t cap, int64_t hammer_iters, int64_t stream_i) {
  int nb = hammer_iters > 0 ? 8 : 1;
  pingpong<<<nb, 1024, 0, (hipStream_t)stream_i>>>(
      (unsigned*)peer, (unsigned*)me, (unsigned long long*)samples,
      (unsigned long long*)rsp_spins, (unsigned*)aborts,
      (int)role, (int)iters, (int)variant, (int)nops,
      (unsigned long long)cap, (int)hammer_iters);
  HIPOK(hipGetLastError());
}
static void run_order(int64_t peer, int64_t me, int64_t arrival, int64_t rsp_spins,
                      int64_t stale,
                      int64_t never, int64_t role, int64_t iters, int64_t dv, int64_t fv,
                      int64_t nops, int64_t cap, int64_t hammer_iters, int64_t stream_i) {
  int nb = hammer_iters > 0 ? 8 : 1;
  order_test<<<nb, 1024, 0, (hipStream_t)stream_i>>>(
      (unsigned*)peer, (unsigned*)me, (unsigned long long*)arrival,
      (unsigned long long*)rsp_spins,
      (unsigned*)stale, (unsigned*)never,
      (int)role, (int)iters, (int)dv, (int)fv, (int)nops,
      (unsigned long long)cap, (int)hammer_iters);
  HIPOK(hipGetLastError());
}
static void run_ll(int64_t peer, int64_t me, int64_t samples, int64_t rsp_spins,
                   int64_t tears, int64_t aborts,
                   int64_t role, int64_t iters, int64_t nops, int64_t cap,
                   int64_t hammer_iters, int64_t stream_i) {
  int nb = hammer_iters > 0 ? 8 : 1;
  ll_test<<<nb, 1024, 0, (hipStream_t)stream_i>>>(
      (unsigned long long*)peer, (unsigned long long*)me, (unsigned long long*)samples,
      (unsigned long long*)rsp_spins,
      (unsigned*)tears, (unsigned*)aborts,
      (int)role, (int)iters, (int)nops, (unsigned long long)cap, (int)hammer_iters);
  HIPOK(hipGetLastError());
}

// ns-per-uncached-peer-read calibration: one wavefront does N dependent reads.
__global__ void read_calib(const unsigned* peer, unsigned long long n, unsigned* sink) {
  unsigned acc = 0;
  for (unsigned long long i = 0; i < n; ++i) acc += load_acq(peer);
  *sink = acc;
}
static void run_read_calib(int64_t peer, int64_t n, int64_t sink, int64_t stream_i) {
  read_calib<<<1, 64, 0, (hipStream_t)stream_i>>>((const unsigned*)peer,
      (unsigned long long)n, (unsigned*)sink);
  HIPOK(hipGetLastError());
}

PYBIND11_MODULE(ar_phase0, m) {
  m.def("alloc_shared", &alloc_shared, py::arg("size"), py::arg("finegrained") = true);
  m.def("open_shared", &open_shared);
  m.def("memzero", &memzero);
  m.def("run_pingpong", &run_pingpong);
  m.def("run_order", &run_order);
  m.def("run_ll", &run_ll);
  m.def("run_read_calib", &run_read_calib);
}
