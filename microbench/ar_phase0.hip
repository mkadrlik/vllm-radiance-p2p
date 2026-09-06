// ar_phase0.hip — Phase 0 microbench: peer-write flush primitives on gfx1100 PCIe.
//
// Measures, on real gfx1100<->gfx1100 fine-grained IPC BAR mappings:
//  (a)  store visibility latency per cache-policy variant (ping-pong RTT),
//  (a2) backoff sweep: RTT vs spin backoff (the starvation/livelock axis),
//  (b)  data-before-flag ordering (the V1 claim under test),
//  (c)  packed {payload,seq} single 8B store: tearing + RTT (LL protocol),
//  (d)  contention: RTT under concurrent in-kernel read-storm blocks,
//  (f)  store x load visibility matrix (which reader sees which writer),
//  (g)  RCCL baseline (driver side, section in phase0_bench.py).
//
// gfx1100 ISA reality (probe2/probe3, verified by compiling kernels that reference
// them): NO shader-readable wall clock — s_timer/s_timerlo/s_timerhi,
// s_memrealtime, clock64() all fail the gfx1100 assembler from a live kernel
// (clang -O0 probes "pass" only because unreferenced functions die before
// assembly). buffer_wbl2 / global_inv: gfx12+/CDNA only.
// Timing = spin-poll counts + wall-time-per-poll derived from the run itself
// (wall_total / sum(spins)), robust to backoff; host wall per iteration reported.
//
// PROTOCOL RULES learned the hard way (rev 1 produced impossible zeros):
//  1. The rendezvous/handshake uses PLAIN volatile stores + plain reads ONLY. It
//     must be immune to the store policy under test, else a broken variant fails
//     the rendezvous and the whole config silently collapses (never=4 pattern).
//  2. Responder results are POSTED INTO THE INITIATOR's scratch (plain stores);
//     rank 1's local host tensors are never readable from rank 0's report.
//  3. Every slot gets its own 64B line (no mixed-policy line sharing).
//  4. Backoff is mandatory in every spin loop (b0 hot spin starves inbound
//     posted writes — that IS the production livelock mechanism, measured here).
//
// SAFETY: every spin bounded by a counter cap sized to <200 ms worst case;
// rendezvous cap 16x that. No unbounded spin exists in this file.
#include <hip/hip_runtime.h>
#include <pybind11/pybind11.h>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
namespace py = pybind11;

#define HIPOK(e) do { hipError_t _e = (e); if (_e != hipSuccess) \
    throw std::runtime_error(std::string("hip: ") + hipGetErrorString(_e)); } while (0)

// store variants (measured leg only — handshake never uses these)
//  0 plain st.global (volatile)
//  1 __hip_atomic_store RELEASE SCOPE_SYS (what V0 uses)
//  2 __threadfence_system() then plain store
//  3 buffer_store sc0 (cached)
//  4 buffer_store glsc (sc1: L2 store, coherent)
//  5 buffer_store sc1 + dependent read-back flush (V3 mechanism)
//  6 plain store + s_waitcnt vmcnt(0) (V1 flush claim)
//  7 plain store + s_waitcnt vmcnt(0),excnt(0) (V1 drain=3)
enum { SV_PLAIN=0, SV_RELSYS=1, SV_TFS=2, SV_BUF_SC0=3, SV_BUF_SC1=4,
       SV_SC1_READFLUSH=5, SV_PLAIN_VMCNT=6, SV_PLAIN_VMEX=7, SV_PACKED=8, SV_COUNT=9 };

#define BUF_GLSC 0x02000000u  // GLSC cache-policy bit (sc1) for raw_buffer builtins
#define MAGIC1 0xABCDEF01u
#define MAGIC2 0xABCDEF02u

// 64B-line slot indices (u32 units). Layout identical for all tests; LL uses the
// u64 view of the same lines (each u64 slot = half a line, fine: single writer
// per u64 slot, and the line belongs to exactly one field at a time).
#define S_REQ    0    // line 0: initiator -> responder (pingpong req / order data)
#define S_RSP    16   // line 1: responder -> initiator (pingpong rsp / order flag)
#define S_ECHO   32   // line 2: order echo / LL mbox
#define S_HSA    48   // line 3: rendezvous A (initiator -> responder), plain
#define S_HSB    64   // line 4: rendezvous B (responder -> initiator), plain
#define S_VIS    80   // line 5: vis_probe tag slot
#define S_RSPT   96   // line 6: responder stats transport (rsp_spins[0..13], never)
#define SCRATCH_U32 128  // 512 B

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

// backoff between spin polls: nops * 63-cycle s_nop.
__device__ __forceinline__ void backoff(int nops) {
  for (int i = 0; i < nops; ++i) asm volatile("s_nop 63" ::: "memory");
}

// policy-immune handshake ops (rule 1): plain volatile everywhere.
__device__ __forceinline__ void hstore(unsigned* p, unsigned v) { *(volatile unsigned*)p = v; }
__device__ __forceinline__ unsigned hload(const unsigned* p) { return *(const volatile unsigned*)p; }
__device__ __forceinline__ void hstore64(unsigned long long* p, unsigned long long v) { *(volatile unsigned long long*)p = v; }
__device__ __forceinline__ unsigned long long hload64(const unsigned long long* p) { return *(const volatile unsigned long long*)p; }

// Hammer: extra workgroups of the SAME kernel storm the peer's scratch with uncached
// reads (GPU_MAX_HW_QUEUES=1 serializes streams; same-kernel workgroups stay concurrent).
__device__ __forceinline__ void hammer_block(const unsigned* w, int hammer_iters, int nops) {
  for (int i = threadIdx.x; i < hammer_iters; i += blockDim.x) {
    (void)load_acq(w); (void)load_acq(w + 1); (void)load_acq(w + 2); (void)load_acq(w + 3);
    backoff(nops);
  }
}

// aborts/never bit layout:
//  bit0 init measured-timeout  bit1 rsp measured-timeout
//  bit2 init rendezvous-fail   bit3 rsp rendezvous-fail
//  bit4 rsp saw DONE sentinel (benign)  bit5 bad magic

#define HB 3  // handshake backoff (cycles bucket -> nops=3 ~ 190 ns) — always nonzero

// ---- (a) ping-pong RTT. role 0 = initiator, 1 = responder. ----
// Initiator samples[] live in initiator device mem (driver reads them).
// Responder posts: rsp_spins[i] (u32) into peer S_RSPT line, and total-miss/never
// into S_RSPT+14 (u32 never|miss), plus hs_spin (rendezvous polls) into S_RSPT+15.
__global__ void pingpong(unsigned* peer_base, unsigned* my_base,
                         unsigned long long* samples,
                         unsigned* aborts,
                         int role, int iters, int variant, int nops,
                         unsigned long long cap, int hammer_iters) {
  if (blockIdx.x != 0) {
    if (hammer_iters > 0) hammer_block(peer_base, hammer_iters, nops);
    return;
  }
  const unsigned long long rcap = cap * 256;  // rendezvous: absorb launch skew + cold start
  if (role == 0) {
    unsigned long long s = 0;
    while (hload(my_base + S_HSB) != MAGIC2) {
      if (++s > rcap) { atomicOr(aborts, 4u); return; }
      if ((s & 255u) == 1u) hstore(peer_base + S_HSA, MAGIC1);  // repost: beat cold-start loss
      backoff(HB);
    }
    for (int i = 1; i <= iters; ++i) {
      store_var(peer_base + S_REQ, (unsigned)i, variant);
      unsigned long long spins = 0;
      bool done = false;
      while (!done) {
        if (load_acq(my_base + S_RSP) >= (unsigned)i) done = true;
        else if (++spins > cap) { atomicOr(aborts, 1u); break; }
        else backoff(nops + HB);
      }
      samples[i - 1] = done ? spins : ~0ull;
      if (!done) break;
    }
    hstore(peer_base + S_HSA, MAGIC1 + 1u);  // DONE sentinel: free the responder
  } else {
    unsigned long long s = 0;
    while (hload(my_base + S_HSA) != MAGIC1) {
      if (++s > rcap) { atomicOr(aborts, 8u); return; }
      backoff(HB);
    }
    hstore(peer_base + S_HSB, MAGIC2);
    int miss = 0;
    hstore(peer_base + S_RSPT + 14, 0u);      // miss count
    hstore(peer_base + S_RSPT + 15, (unsigned)s);  // rendezvous polls
    for (int i = 1; i <= iters; ++i) {
      unsigned long long spins = 0;
      bool ok = false;
      while (!ok) {
        if (hload(my_base + S_REQ) >= (unsigned)i) ok = true;
        else if (hload(my_base + S_HSA) == MAGIC1 + 1u) { atomicOr(aborts, 16u); goto pp_out; }
        else if (++spins > cap) break;
        else backoff(nops + HB);
      }
      if (!ok) {
        atomicOr(aborts, 2u);
        hstore(peer_base + S_RSPT + 14, (unsigned)(++miss));
        if (miss > 3) break;
        continue;
      }
      miss = 0;
      if (i < 14) hstore(peer_base + S_RSPT + i, (unsigned)spins);  // first 13 samples
      store_var(peer_base + S_RSP, (unsigned)i, variant);
    }
  pp_out:;
  }
}

// ---- (b) ordering: data posted with dv then flag with fv; responder checks the
// payload when the flag arrives. stale>0 => flag outran data (flush failed).
// Responder posts: stale count -> S_RSPT+14, arrival spins[i] -> S_RSPT lines.
__global__ void order_test(unsigned* peer_base, unsigned* my_base,
                           unsigned long long* arrival,
                           unsigned* stale, unsigned* flags_never,
                           int role, int iters,
                           int data_variant, int flag_variant, int nops,
                           unsigned long long cap, int hammer_iters) {
  if (blockIdx.x != 0) {
    if (hammer_iters > 0) hammer_block(peer_base, hammer_iters, nops);
    return;
  }
  const unsigned long long rcap = cap * 16;
  if (role == 0) {
    hstore(peer_base + S_HSA, MAGIC1);
    unsigned long long s = 0;
    while (hload(my_base + S_HSB) != MAGIC2) {
      if (++s > rcap) { atomicOr(flags_never, 4u); return; }
      backoff(HB);
    }
    for (int i = 1; i <= iters; ++i) {
      store_var(peer_base + S_REQ, 0xA5000000u | (unsigned)i, data_variant);
      store_var(peer_base + S_RSP, (unsigned)i, flag_variant);
      unsigned long long spins = 0;
      bool done = false;
      while (!done) {
        if (hload(my_base + S_ECHO) >= (unsigned)i) done = true;
        else if (++spins > cap) { atomicOr(flags_never, 1u); break; }
        else backoff(nops + HB);
      }
      arrival[i - 1] = done ? spins : ~0ull;
      if (!done) break;
    }
    hstore(peer_base + S_HSA, MAGIC1 + 1u);
  } else {
    unsigned long long s = 0;
    while (hload(my_base + S_HSA) != MAGIC1) {
      if (++s > rcap) { atomicOr(flags_never, 8u); return; }
      backoff(HB);
    }
    hstore(peer_base + S_HSB, MAGIC2);
    int miss = 0;
    unsigned stale_n = 0, fresh_n = 0;
    for (int i = 1; i <= iters; ++i) {
      unsigned long long spins = 0;
      bool ok = false;
      while (!ok) {
        if (hload(my_base + S_RSP) >= (unsigned)i) ok = true;
        else if (hload(my_base + S_HSA) == MAGIC1 + 1u) { atomicOr(flags_never, 16u); goto ord_out; }
        else if (++spins > cap) break;
        else backoff(nops + HB);
      }
      if (!ok) {
        atomicOr(flags_never, 2u);
        if (++miss > 3) break;
        continue;
      }
      miss = 0;
      unsigned d = hload(my_base + S_REQ);
      bool fresh = (d == (0xA5000000u | (unsigned)i));
      if (fresh) ++fresh_n; else ++stale_n;
      // transport (u32 view of S_RSPT line): u64 samples occupy u32 0..13
      // (7 slots); stale -> +14, fresh -> +15 (no overlap).
      if (i < 7)
        hstore64((unsigned long long*)(peer_base + S_RSPT) + i,
                 spins | (fresh ? (1ull << 63) : 0ull));
      hstore(peer_base + S_RSPT + 14, stale_n);
      hstore(peer_base + S_RSPT + 15, fresh_n);
      hstore(peer_base + S_ECHO, (unsigned)i);
    }
  ord_out:;
  }
}

// ---- (c) LL packed 8B: mbox line 2 (u64 slots 4,5), echo line 1 (u64 2,3). ----
// Handshake uses the plain u32 slots (policy-immune).
__global__ void ll_test(unsigned long long* peer_base, unsigned long long* my_base,
                        unsigned long long* samples,
                        unsigned* tears, unsigned* aborts,
                        int role, int iters, int nops, unsigned long long cap,
                        int hammer_iters) {
  if (blockIdx.x != 0) {
    if (hammer_iters > 0) hammer_block((const unsigned*)peer_base, hammer_iters, nops);
    return;
  }
  const unsigned PAY = 0xBEEF0000u;
  const unsigned long long rcap = cap * 16;
  unsigned* peer32 = (unsigned*)peer_base;
  unsigned* my32 = (unsigned*)my_base;
  const int MBOX = 4, ECHO = 2;  // u64 slot indices (lines 2 and 1)
  if (role == 0) {
    hstore(peer32 + S_HSA, MAGIC1);
    unsigned long long s = 0;
    while (hload(my32 + S_HSB) != MAGIC2) {
      if (++s > rcap) { atomicOr(aborts, 4u); return; }
      backoff(HB);
    }
    for (int i = 1; i <= iters; ++i) {
      store_packed(peer_base + MBOX, PAY ^ (unsigned)i, (unsigned)i);
      unsigned long long spins = 0;
      bool done = false;
      unsigned long long v = 0;
      while (!done) {
        v = load_acq64(my_base + ECHO);
        if ((v >> 32) >= (unsigned)i) done = true;
        else if (++spins > cap) { atomicOr(aborts, 1u); break; }
        else backoff(nops + HB);
      }
      if (!done) { samples[i - 1] = ~0ull; break; }
      if ((unsigned)v != (PAY ^ (unsigned)i)) atomicAdd(tears, 1u);
      samples[i - 1] = spins;
    }
    hstore(peer32 + S_HSA, MAGIC1 + 1u);
  } else {
    unsigned long long s = 0;
    while (hload(my32 + S_HSA) != MAGIC1) {
      if (++s > rcap) { atomicOr(aborts, 8u); return; }
      backoff(HB);
    }
    hstore(peer32 + S_HSB, MAGIC2);
    int miss = 0;
    unsigned rsp_max = 0;
    for (int i = 1; i <= iters; ++i) {
      unsigned long long spins = 0;
      bool ok = false;
      while (!ok) {
        unsigned long long v = hload64(my_base + MBOX);
        if ((v >> 32) >= (unsigned)i) ok = true;
        else if (hload(my32 + S_HSA) == MAGIC1 + 1u) { atomicOr(aborts, 16u); goto ll_out; }
        else if (++spins > cap) break;
        else backoff(nops + HB);
      }
      if (!ok) {
        atomicOr(aborts, 2u);
        if (++miss > 3) break;
        continue;
      }
      miss = 0;
      unsigned long long v = hload64(my_base + MBOX);
      if ((unsigned)spins > rsp_max) rsp_max = (unsigned)spins;
      store_packed(peer_base + ECHO, (unsigned)v, (unsigned)i);
      hstore(peer32 + S_RSPT + 14, rsp_max);  // max responder-side spin seen
    }
  ll_out:;
  }
}

// ---- (f) visibility matrix: can reader variant rv SEE writer variant sv? ----
// One-shot post (no repost race): initiator posts tag ONCE, responder polls with
// plain AND variant loads? No — responder polls with rv only; the tag is unique so
// no false positives. Initiator reposts every ~1 us to beat launch skew; responder
// records first-hit spin count. Result transported via plain store into S_VIS+16.
__device__ __forceinline__ unsigned load_var(const unsigned* p, int variant) {
  switch (variant) {
    case 0: return *(const volatile unsigned*)p;                       // plain nc read
    case 1: return load_acq(p);                                        // atomic SCOPE_SYS
    case 2: return buf_load32<0>(p);                                   // buffer sc0
    case 3: return buf_load32<BUF_GLSC>(p);                            // buffer sc1
    case 4: { unsigned v = buf_load32<BUF_GLSC>(p);                    // sc1 + wait
              asm volatile("s_waitcnt vmcnt(0)" ::: "memory"); return v; }
    default: return load_acq(p);
  }
}
#define LD_COUNT 5

__global__ void vis_probe(unsigned* peer_base, unsigned* my_base,
                          unsigned* dummy, unsigned* ok_bits,
                          int role, int sv, int rv, unsigned long long cap, unsigned tag) {
  // responder polls up to cap ~1 us-polls (~100 ms window)
  if (role == 0) {
    hstore(my_base + S_VIS + 16, 0xFFFFFFFFu);  // clear result slot (local line)
    unsigned long long s = 0;
    while (s < cap) { store_var(peer_base + S_VIS, tag, sv); ++s; backoff(15); }
    // wait for the responder's plain-store result (posted into MY line)
    unsigned long long w = 0;
    while (hload(my_base + S_VIS + 16) == 0xFFFFFFFFu) {
      if (++w > cap * 4) { *ok_bits = 0u; return; }  // responder gone
      backoff(15);
    }
    *ok_bits = hload(my_base + S_VIS + 16) == 1u ? 1u : 0u;
  } else {
    unsigned long long s = 0;
    bool seen = false;
    while (true) {
      if (load_var(my_base + S_VIS, rv) == tag) { seen = true; break; }
      if (++s > cap) break;
      backoff(15);
    }
    hstore(peer_base + S_VIS + 16, seen ? 1u : 0u);
  }
}

// ---- (h) delivery under production-shape spin (the livelock reproduction) ----
// Reader polls ITS OWN line my[S_VIS] with W concurrent wavefronts (threadIdx stride
// W*64 lanes) — the same shape as the production kernel's 24 blocks x 1024 thr
// polling flags — with backoff b. Writer repeatedly posts a fresh tag to the peer's
// S_VIS line (plain volatile store, like production data+flag path). Question:
// does a HOT multi-wavefront spin starve inbound posted writes (delivery time ->
// timeout), and does backoff bound it? This is the V0 serving-livelock hypothesis,
// isolated from CUDA graphs.
// Result transport: reader posts elapsed spin-iterations-to-see into peer S_RSPT+14,
// writer's host wall covers launch->first-observe via a plain done flag (S_RSPT+15=1).
__global__ void deliver_test(unsigned* peer_base, unsigned* my_base,
                             int role, int b, unsigned long long cap, unsigned tag) {
  if (role == 0) {
    // writer: post fresh tags until the reader confirms (bounded)
    unsigned long long s = 0;
    hstore(my_base + S_RSPT + 15, 0u);
    while (hload(my_base + S_RSPT + 15) == 0u) {
      hstore(peer_base + S_VIS, tag + (unsigned)(s & 0xFFFFu));
      ++s;
      if (s > cap) { hstore(my_base + S_RSPT + 13, 0xFFFFFFFFu); return; }
      backoff(15);          // ~1 us between reposts
    }
    hstore(my_base + S_RSPT + 13, (unsigned)s);  // reposts until confirm (~us latency)
  } else {
    // reader: blockDim threads (all wavefronts) poll the SAME line — exactly the
    // production flag shape (24 blocks x 1024 thr on one 64B line). Wave 0 lane 0
    // records the poll-round count.
    unsigned long long s = 0;
    volatile const unsigned* line = (volatile const unsigned*)(my_base + S_VIS);
    bool seen = false;
    while (!seen) {
      unsigned v = *line;
      if (v >= tag) seen = true;
      else if (++s > cap) break;
      else backoff(b);
    }
    if (threadIdx.x == 0) {
      hstore(peer_base + S_RSPT + 14, seen ? (unsigned)s : 0xFFFFFFFFu);
      hstore(peer_base + S_RSPT + 15, 1u);  // confirm to writer
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
static void read_words(int64_t p, int64_t n, int64_t out) {
  HIPOK(hipMemcpy((void*)out, (void*)p, (size_t)n * 4, hipMemcpyDeviceToDevice));
}

static void run_pingpong(int64_t peer, int64_t me, int64_t samples,
                         int64_t aborts, int64_t role, int64_t iters, int64_t variant,
                         int64_t nops, int64_t cap, int64_t hammer_iters, int64_t stream_i) {
  int nb = hammer_iters > 0 ? 8 : 1;
  pingpong<<<nb, 1024, 0, (hipStream_t)stream_i>>>(
      (unsigned*)peer, (unsigned*)me, (unsigned long long*)samples, (unsigned*)aborts,
      (int)role, (int)iters, (int)variant, (int)nops,
      (unsigned long long)cap, (int)hammer_iters);
  HIPOK(hipGetLastError());
}
static void run_order(int64_t peer, int64_t me, int64_t arrival,
                      int64_t stale, int64_t never, int64_t role, int64_t iters,
                      int64_t dv, int64_t fv, int64_t nops, int64_t cap,
                      int64_t hammer_iters, int64_t stream_i) {
  int nb = hammer_iters > 0 ? 8 : 1;
  order_test<<<nb, 1024, 0, (hipStream_t)stream_i>>>(
      (unsigned*)peer, (unsigned*)me, (unsigned long long*)arrival,
      (unsigned*)stale, (unsigned*)never,
      (int)role, (int)iters, (int)dv, (int)fv, (int)nops,
      (unsigned long long)cap, (int)hammer_iters);
  HIPOK(hipGetLastError());
}
static void run_ll(int64_t peer, int64_t me, int64_t samples,
                   int64_t tears, int64_t aborts,
                   int64_t role, int64_t iters, int64_t nops, int64_t cap,
                   int64_t hammer_iters, int64_t stream_i) {
  int nb = hammer_iters > 0 ? 8 : 1;
  ll_test<<<nb, 1024, 0, (hipStream_t)stream_i>>>(
      (unsigned long long*)peer, (unsigned long long*)me, (unsigned long long*)samples,
      (unsigned*)tears, (unsigned*)aborts,
      (int)role, (int)iters, (int)nops, (unsigned long long)cap, (int)hammer_iters);
  HIPOK(hipGetLastError());
}
static void run_vis_probe(int64_t peer, int64_t me, int64_t dummy, int64_t ok,
                          int64_t role, int64_t sv, int64_t rv, int64_t cap,
                          int64_t tag, int64_t stream_i) {
  vis_probe<<<1, 64, 0, (hipStream_t)stream_i>>>(
      (unsigned*)peer, (unsigned*)me, (unsigned*)dummy, (unsigned*)ok,
      (int)role, (int)sv, (int)rv, (unsigned long long)cap, (unsigned)tag);
  HIPOK(hipGetLastError());
}

static void run_deliver(int64_t peer, int64_t me, int64_t role, int64_t b,
                        int64_t cap, int64_t tag, int64_t nthreads, int64_t stream_i) {
  deliver_test<<<1, (int)nthreads, 0, (hipStream_t)stream_i>>>(
      (unsigned*)peer, (unsigned*)me, (int)role, (int)b,
      (unsigned long long)cap, (unsigned)tag);
  HIPOK(hipGetLastError());
}

// ns-per-poll calibration: one wavefront does N iterations of the SAME loop shape as
// the spin loops (one uncached local fine-grained read + backoff(nops)).
__global__ void read_calib(const unsigned* word, unsigned long long n, int nops, unsigned* sink) {
  unsigned acc = 0;
  for (unsigned long long i = 0; i < n; ++i) {
    acc += load_acq(word);
    backoff(nops);
  }
  *sink = acc;
}
static void run_read_calib(int64_t word, int64_t n, int64_t nops, int64_t sink, int64_t stream_i) {
  read_calib<<<1, 64, 0, (hipStream_t)stream_i>>>((const unsigned*)word,
      (unsigned long long)n, (int)nops, (unsigned*)sink);
  HIPOK(hipGetLastError());
}

PYBIND11_MODULE(ar_phase0, m) {
  m.def("alloc_shared", &alloc_shared, py::arg("size"), py::arg("finegrained") = true);
  m.def("open_shared", &open_shared);
  m.def("memzero", &memzero);
  m.def("read_words", &read_words);
  m.def("run_pingpong", &run_pingpong);
  m.def("run_order", &run_order);
  m.def("run_ll", &run_ll);
  m.def("run_read_calib", &run_read_calib);
  m.def("run_vis_probe", &run_vis_probe);
  m.def("run_deliver", &run_deliver);
}
