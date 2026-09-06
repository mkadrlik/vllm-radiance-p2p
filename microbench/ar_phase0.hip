// ar_phase0.hip — Phase 0 microbench: peer-write flush primitives on gfx1100 PCIe P2P.
//
// Measures, on real gfx1100<->gfx1100 fine-grained IPC BAR mappings:
//  (a) store visibility latency per cache-policy variant (ping-pong RTT),
//  (b) data-before-flag ordering: does flush variant X make a payload observable
//      with bounded latency before its flag (the V1 claim under test),
//  (c) packed {payload,seq} single 8B store: tearing + RTT distribution,
//  (d) contention: RTT under a concurrent peer-spin hammer with s_nop backoff levels.
//
// SAFETY: every spin is bounded by a globaltimer abort (never unbounded). Nothing
// here can wedge a GPU beyond the abort window. No vLLM, no prod containers touched.
#include <hip/hip_runtime.h>
#include <pybind11/pybind11.h>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
namespace py = pybind11;

#define HIPOK(e) do { hipError_t _e = (e); if (_e != hipSuccess) \
    throw std::runtime_error(std::string("hip: ") + hipGetErrorString(_e)); } while (0)

// store variants (both legs of ping-pong / flag+data posts use this)
//  0 plain st.global (default policy)
//  1 __hip_atomic_store RELEASE SCOPE_SYS (what V0 uses)
//  2 __threadfence_system() then plain store
//  3 buffer_store, cache policy 0 (sc0/cached)
//  4 buffer_store glsc (sc1)
//  5 buffer_store glsc + s_buffer_wbl2 + waits
//  6 plain store + s_waitcnt vmcnt(0) (V1 flush claim)
//  7 plain store + s_waitcnt vmcnt(0),excnt(0) (V1 drain=3)
enum { SV_PLAIN=0, SV_RELSYS=1, SV_TFS=2, SV_BUF_SC0=3, SV_BUF_SC1=4,
       SV_SC1_READFLUSH=5, SV_PLAIN_VMCNT=6, SV_PLAIN_VMEX=7, SV_PACKED=8, SV_COUNT=9 };

__device__ __forceinline__ unsigned long long gt() {
  // gfx1100: s_timer is 1 MHz (too coarse), s_memrealtime is gfx12+. clock64()
  // (per-SHADER_CORE counter) compiles for gfx1100 and is monotonic within a
  // kernel — all a same-GPU RTT delta needs. Tick rate calibrated host-side.
  return clock64();
}
__device__ __forceinline__ void timer_init() {}  // no-op; kept for call-site symmetry

// aux cache-policy bits for __builtin_amdgcn_raw_buffer_*: GLSC = 1<<25 (sc1).
#define BUF_GLSC 0x02000000u

template <unsigned AUX>
__device__ __forceinline__ void buf_store32(unsigned* p, unsigned v) {
  auto rsrc = __builtin_amdgcn_make_buffer_rsrc((void*)p, 0, 0xFFFFFFFFu, 0);
  __builtin_amdgcn_raw_buffer_store_b32(v, rsrc, 0, 0, AUX);
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
      // gfx1100 HAS NO buffer_wbl2 / global_inv / s_memrealtime (all gfx12+/CDNA).
      // Nearest L2-forcing primitive: dependent non-posted READ from the peer —
      // PCIe ordering sends the read after the posted write completes at target.
      // This is the V3 "read-back flush" mechanism, measured here for latency.
      buf_store32<BUF_GLSC>(p, v);
      asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
      {
        auto rsrc = __builtin_amdgcn_make_buffer_rsrc((void*)p, 0, 0xFFFFFFFFu, 0);
        volatile unsigned rb = __builtin_amdgcn_raw_buffer_load_b32(rsrc, 0, 0, BUF_GLSC);
        asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
        (void)rb;
      }
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

// packed 8-byte store for the LL test: {payload:32, seq:32} in one aligned u64.
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

// backoff between spin polls: nops * 63-cycle s_nop (max immediate 63), 0 = hot spin.
__device__ __forceinline__ void backoff(int nops) {
  for (int i = 0; i < nops; ++i) asm volatile("s_nop 63" ::: "memory");
}

// Layout of the ping-pong scratch (u32 indices):
//  [0]=req flag (starter -> responder), [1]=rsp flag (responder -> starter)
//  [2..]=unused. Allocated fine-grained IPC, 64B apart via u64 slots to avoid line sharing.
struct Slots { volatile unsigned long long* req; volatile unsigned long long* rsp; };

// ---- ping-pong RTT kernel (a). role 0 = starter (records RTT), 1 = responder. ----
// Contention mode (d): blocks 1..nb-1 are local read-storm hammers on the PEER's
// scratch (uncached reads, backoff nops) — same kernel => same-wavefront-schedule,
// concurrent workgroups even under GPU_MAX_HW_QUEUES=1 (a side-stream kernel would
// serialize on the single HW queue and produce a fake uncontended result).
__global__ void pingpong(unsigned long long* peer_base, unsigned long long* my_base,
                         unsigned long long* samples, unsigned* aborts,
                         int role, int iters, int variant, int nops,
                         unsigned long long abort_ns, int hammer_iters) {
  timer_init();
  if (blockIdx.x != 0) {
    if (hammer_iters > 0) {
      const unsigned* w = (const unsigned*)peer_base;
      for (int i = threadIdx.x; i < hammer_iters; i += blockDim.x) {
        (void)load_acq(w); (void)load_acq(w + 1); (void)load_acq(w + 2); (void)load_acq(w + 3);
        backoff(nops);
      }
    }
    return;
  }
  unsigned long long* peer_req = peer_base + 0;
  unsigned long long* my_rsp   = my_base + 8;   // byte offset 8 => u64 slot 1
  unsigned long long* peer_rsp = peer_base + 8;
  unsigned long long* my_req   = my_base + 0;
  if (role == 0) {
    for (int i = 1; i <= iters; ++i) {
      unsigned long long t0 = gt();
      store_var((unsigned*)peer_req, (unsigned)i, variant);
      bool done = false;
      while (!done) {
        if (load_acq((const unsigned*)my_rsp) >= (unsigned)i) done = true;
        else if (gt() - t0 > abort_ns) { atomicOr(aborts, 1u); break; }
        else backoff(nops);
      }
      samples[i - 1] = done ? (gt() - t0) : ~0ull;
    }
  } else {
    for (int i = 1; i <= iters; ++i) {
      unsigned long long t0 = gt();
      bool done = false;
      while (!done) {
        if (load_acq((const unsigned*)my_req) >= (unsigned)i) done = true;
        else if (gt() - t0 > abort_ns * 4) { atomicOr(aborts, 2u); break; }
        else backoff(nops);
      }
      if (!done) break;  // peer stalled: exit now, never cascade timeouts
      store_var((unsigned*)peer_rsp, (unsigned)i, variant);
    }
  }
}

// ---- ordering test (b): writer posts data then flag; reader verifies payload when
// flag arrives; measures stale-data count per (data_variant, flag_variant) and RTT.
// Echo backpressure: writer waits for reader's echo before the next iteration, so
// RTT is wire latency + check, not poll granularity.
// scratch u32 layout: [0]=data, [2]=flag, [8]=echo
__global__ void order_test(unsigned* peer_base, unsigned* my_base,
                           unsigned long long* arrival_ns, unsigned* stale,
                           unsigned* flags_never, int role, int iters,
                           int data_variant, int flag_variant, int nops,
                           unsigned long long abort_ns) {
  timer_init();
  if (role == 0) {
    for (int i = 1; i <= iters; ++i) {
      unsigned long long t0 = gt();
      unsigned payload = 0xA5000000u | (unsigned)i;
      store_var(peer_base + 0, payload, data_variant);
      store_var(peer_base + 2, (unsigned)i, flag_variant);
      bool done = false;
      while (!done) {
        if (load_acq(my_base + 8) >= (unsigned)i) done = true;
        else if (gt() - t0 > abort_ns) { atomicOr(flags_never, 1u); break; }
        else backoff(nops);
      }
      if (!done) break;  // peer stalled: exit now, never cascade timeouts
      arrival_ns[i - 1] = done ? (gt() - t0) : ~0ull;
    }
  } else {
    for (int i = 1; i <= iters; ++i) {
      unsigned long long t0 = gt();
      unsigned f = 0;
      bool rdone = false;
      while (!rdone) {
        f = load_acq(my_base + 2);
        if (f >= (unsigned)i) rdone = true;
        else if (gt() - t0 > abort_ns * 4) { atomicOr(flags_never, 2u); break; }
        else backoff(nops);
      }
      if (!rdone) break;
      unsigned d = load_acq(my_base + 0);
      if (d != (0xA5000000u | (unsigned)i)) atomicAdd(stale, 1u);
      store_var(peer_base + 8, (unsigned)i, flag_variant);
    }
  }
}

// ---- LL packed test (c): u64 {payload,seq} single store; tearing + RTT ----
__global__ void ll_test(unsigned long long* peer_base, unsigned long long* my_base,
                        unsigned long long* samples, unsigned* tears, unsigned* aborts,
                        int role, int iters, int nops, unsigned long long abort_ns,
                        int hammer_iters) {
  timer_init();
  if (blockIdx.x != 0) {
    if (hammer_iters > 0) {
      const unsigned* w = (const unsigned*)peer_base;
      for (int i = threadIdx.x; i < hammer_iters; i += blockDim.x) {
        (void)load_acq(w); (void)load_acq(w + 1); (void)load_acq(w + 2); (void)load_acq(w + 3);
        backoff(nops);
      }
    }
    return;
  }
  if (role == 0) {
    for (int i = 1; i <= iters; ++i) {
      unsigned long long t0 = gt();
      store_packed(peer_base + 16, (unsigned)(0xBEEF0000u ^ (unsigned)i), (unsigned)i);
      // RTT = wait for the responder's echo (its packed reply on OUR slot +24)
      bool done = false;
      unsigned long long v = 0;
      while (!done) {
        v = load_acq64(my_base + 24);
        if ((v >> 32) >= (unsigned)i) done = true;
        else if (gt() - t0 > abort_ns) { atomicOr(aborts, 1u); break; }
        else backoff(nops);
      }
      if (done) {
        unsigned long long w = load_acq64(my_base + 16);
        if ((unsigned)(w & 0xFFFFFFFFu) != (0xBEEF0000u ^ (unsigned)i)) atomicAdd(tears, 1u);
        samples[i - 1] = gt() - t0;
      } else samples[i - 1] = ~0ull;
    }
  } else {
    // responder echoes back packed pair on slot +24 so starter can ping-pong too
    for (int i = 1; i <= iters; ++i) {
      unsigned long long t0 = gt();
      bool done = false;
      while (!done) {
        if ((load_acq64(my_base + 16) >> 32) >= (unsigned)i) done = true;
        else if (gt() - t0 > abort_ns * 4) { atomicOr(aborts, 2u); break; }
        else backoff(nops);
      }
      if (done) store_packed(peer_base + 24, (unsigned)i, (unsigned)i);
    }
  }
}

// ---- clock offset calibration: both ranks stamp their gt at (near) the same wall time.
__global__ void stamp(unsigned long long* out) { *out = gt(); }

// s_timer tick-rate calibration: gt0, ~1 s busy delay, gt1. Ticks/1e9ns = rate.
// (gfx1100 has no __nanosleep; use a coarse delay loop — rate is what we measure.)
__global__ void calib(unsigned long long* out) {
  timer_init();
  unsigned long long t0 = gt();
  volatile unsigned sink = 0;
  for (unsigned long long i = 0; i < 200000000ull; ++i) sink += i;  // ~1 s
  unsigned long long t1 = gt();
  out[0] = t0; out[1] = t1; out[2] = sink;
}

// ---- one-way peer-store visibility: seq+timestamp on one u64, reader arrival stamp.
// u64 slots: [0]=mbox (starter->reader), [1]=echo (reader->starter).
// value = (seq<<32) | (gt_low32 at write). Reader records (gt_low32 at observe - arrival_base).
// Starter also records RTT (echo wait) so one-way ≈ RTT/2 sanity check.
__global__ void oneway(unsigned long long* peer_base, unsigned long long* my_base,
                       unsigned long long* rtt, unsigned long long* arrive,
                       unsigned* aborts, int role, int iters, int variant, int nops,
                       unsigned long long abort_ns) {
  timer_init();
  if (role == 0) {
    for (int i = 1; i <= iters; ++i) {
      unsigned long long t0 = gt();
      unsigned tag = (unsigned)(t0 & 0xFFFFFFFFu);
      // single 8B store carrying seq+timestamp (no separate flush needed if LL holds)
      store_packed((unsigned long long*)peer_base, tag, (unsigned)i);
      unsigned long long v = 0;
      while ((((v = load_acq64(my_base + 8)) >> 32)) < (unsigned)i) {
        if (gt() - t0 > abort_ns) { atomicOr(aborts, 1u); break; }
        backoff(nops);
      }
      rtt[i - 1] = (v >> 32 >= (unsigned)i) ? (gt() - t0) : ~0ull;
    }
  } else {
    for (int i = 1; i <= iters; ++i) {
      unsigned long long t0 = gt();
      unsigned long long v = 0;
      while ((((v = load_acq64(my_base)) >> 32)) < (unsigned)i) {
        if (gt() - t0 > abort_ns * 4) { atomicOr(aborts, 2u); break; }
        backoff(nops);
      }
      if (v >> 32 >= (unsigned)i) arrive[i - 1] = gt() - t0;  // spin residual (poll granularity)
      store_packed((unsigned long long*)(peer_base + 8), 0u, (unsigned)i);
    }
  }
}

// ---- contention hammer (d): spin-read a peer word, ITERATION-bound (no s_timer —
// it's a single per-GPU register and timer_init in a concurrent kernel would reset
// the measurement clock). Calibrated via run-rate in the driver.
__global__ void hammer(const unsigned* peer_word, unsigned long long iters, int nops) {
  for (unsigned long long i = 0; i < iters; ++i) {
    (void)load_acq(peer_word);
    (void)load_acq(peer_word + 1);
    (void)load_acq(peer_word + 2);
    (void)load_acq(peer_word + 3);
    backoff(nops);
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

static void run_pingpong(int64_t peer, int64_t me, int64_t samples, int64_t aborts,
                         int64_t role, int64_t iters, int64_t variant, int64_t nops,
                         int64_t abort_ns, int64_t hammer_iters, int64_t stream_i) {
  int nb = hammer_iters > 0 ? 8 : 1;
  pingpong<<<nb, 1024, 0, (hipStream_t)stream_i>>>(
      (unsigned long long*)peer, (unsigned long long*)me,
      (unsigned long long*)samples, (unsigned*)aborts,
      (int)role, (int)iters, (int)variant, (int)nops, (unsigned long long)abort_ns,
      (int)hammer_iters);
  HIPOK(hipGetLastError());
}
static void run_order(int64_t peer, int64_t me, int64_t arrival, int64_t stale,
                      int64_t never, int64_t role, int64_t iters, int64_t dv, int64_t fv,
                      int64_t nops, int64_t abort_ns, int64_t stream_i) {
  order_test<<<1, 64, 0, (hipStream_t)stream_i>>>(
      (unsigned*)peer, (unsigned*)me, (unsigned long long*)arrival,
      (unsigned*)stale, (unsigned*)never,
      (int)role, (int)iters, (int)dv, (int)fv, (int)nops, (unsigned long long)abort_ns);
  HIPOK(hipGetLastError());
}
static void run_ll(int64_t peer, int64_t me, int64_t samples, int64_t tears, int64_t aborts,
                   int64_t role, int64_t iters, int64_t nops, int64_t abort_ns,
                   int64_t hammer_iters, int64_t stream_i) {
  int nb = hammer_iters > 0 ? 8 : 1;
  ll_test<<<nb, 1024, 0, (hipStream_t)stream_i>>>(
      (unsigned long long*)peer, (unsigned long long*)me, (unsigned long long*)samples,
      (unsigned*)tears, (unsigned*)aborts,
      (int)role, (int)iters, (int)nops, (unsigned long long)abort_ns, (int)hammer_iters);
  HIPOK(hipGetLastError());
}
static void run_stamp(int64_t out, int64_t stream_i) {
  stamp<<<1, 1, 0, (hipStream_t)stream_i>>>((unsigned long long*)out);
  HIPOK(hipGetLastError());
}
static void run_calib(int64_t out, int64_t stream_i) {
  calib<<<1, 1, 0, (hipStream_t)stream_i>>>((unsigned long long*)out);
  HIPOK(hipGetLastError());
}
static void run_oneway(int64_t peer, int64_t me, int64_t rtt, int64_t arrive, int64_t aborts,
                       int64_t role, int64_t iters, int64_t nops, int64_t abort_ns, int64_t stream_i) {
  oneway<<<1, 64, 0, (hipStream_t)stream_i>>>(
      (unsigned long long*)peer, (unsigned long long*)me, (unsigned long long*)rtt,
      (unsigned long long*)arrive, (unsigned*)aborts,
      (int)role, (int)iters, 8, (int)nops, (unsigned long long)abort_ns);
  HIPOK(hipGetLastError());
}
static void run_hammer(int64_t peer_word, int64_t duration_ns, int64_t nops, int64_t blocks, int64_t stream_i) {
  hammer<<<(int)blocks, 1024, 0, (hipStream_t)stream_i>>>((const unsigned*)peer_word,
      (unsigned long long)duration_ns, (int)nops);
  HIPOK(hipGetLastError());
}

PYBIND11_MODULE(ar_phase0, m) {
  m.def("alloc_shared", &alloc_shared, py::arg("size"), py::arg("finegrained") = true);
  m.def("open_shared", &open_shared);
  m.def("memzero", &memzero);
  m.def("run_pingpong", &run_pingpong);
  m.def("run_order", &run_order);
  m.def("run_ll", &run_ll);
  m.def("run_stamp", &run_stamp);
  m.def("run_calib", &run_calib);
  m.def("run_oneway", &run_oneway);
  m.def("run_hammer", &run_hammer);
}
