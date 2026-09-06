// ar_proto.hip — Phase 1 prototypes: replay-safe one-shot all-reduce, two variants.
//
// TOPOLOGY CHOICE (forced by Phase 0, docs/PHASE0_RESULTS.md): the only spin
// topology that delivered reliably on gfx1100 PCIe is a SINGLE queue region with
// ONE consumer block polling with ~2000-cycle backoff. Production V0's 24-block
// per-chunk flags each hot-spinning uncached reads is exactly the shape that
// starved inbound posted writes. One block x 1024 threads also covers the decode
// bandwidth budget (Phase 0: RTT p50 ~22 us at b10000; push of 640 KB on one block
// ~130 us PCIe-bound — acceptable for the prototype; multi-block push is Phase-2
// tuning, gated on the same single-flag protocol with a device barrier).
//
// F) flush variant: block 0 pushes the input as plain int4 global stores (16B,
//    vmcnt-flushed — buffer_store NEVER delivers on gfx1100), posts ONE seq flag
//    (plain store), spins on its own flag with backoff, reduces locally.
// L) LL variant: block 0 pushes 8B {payload32, seq32} words — no separate flag;
//    every word is self-validating (tag==seq => data+arrival proof in one TLP).
//
// Both: bounded spin (SPIN_CAP); on timeout NEVER reduce — set a host-mapped
// poison word (the recovery hook the serving path lacks). seq is a device-resident
// counter (graph-replay safe, as in V0); scratch double-buffered by parity.
//
// use_hz=1 (microbench eager path): block-0 device rendezvous with phase-flipping
// magics absorbs kernel-launch skew without clear-races and without host cost.
// use_hz=0 (production graph replay): ranks replay in lockstep; skip the handshake.
#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include <hip/hip_fp16.h>
#include <pybind11/pybind11.h>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
namespace py = pybind11;

#define HIPOK(e) do { hipError_t _e = (e); if (_e != hipSuccess) \
    throw std::runtime_error(std::string("hip: ") + hipGetErrorString(_e)); } while (0)
#define SPIN_CAP 5000000ull  // ~5 s ceiling at ~1 us/poll (absorbs queueing behind prod)
#define BACKOFF 31           // 31 x s_nop 63 ≈ 2000 cycles — Phase-0 sweet spot
#define MAGIC1 0xABCDEF01u
#define MAGIC2 0xABCDEF02u

// flags layout (u32 indices; alloc >= 8KB):
//   [256] F_SEQ data flag (line 16)
//   [272] F_SEEN diagnostic, [288..] stages (line 17,18)
//   handshake rings: rank0 posts hs_a[s&63] (lines 32..95), rank1 echoes
//   hs_b[s&63] (lines 96..159). Every slot owns its 64B line; no two writers
//   share a line; handshake values never satisfy the data flag's >= test.
#define F_SEQ 256
#define F_SEEN 272
#define F_HS_A(s) (32 * 16 + 16 * ((s) & 63u))    // lines 32..95
#define F_HS_B(s) (96 * 16 + 16 * ((s) & 63u))    // lines 96..159

__device__ __forceinline__ void bspin_backoff() {
  for (int i = 0; i < BACKOFF; ++i) asm volatile("s_nop 63" ::: "memory");
}
__device__ __forceinline__ void store_plain(unsigned* p, unsigned v) {
  *(volatile unsigned*)p = v;
  asm volatile("s_waitcnt vmcnt(0) expcnt(0)" ::: "memory");
}
__device__ __forceinline__ unsigned load_plain(const unsigned* p) {
  return *(const volatile unsigned*)p;
}
__device__ __forceinline__ unsigned load_acq(const unsigned* p) {
  return __hip_atomic_load(p, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM);
}
__device__ __forceinline__ unsigned long long load_acq64(const unsigned long long* p) {
  return __hip_atomic_load(p, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM);
}
__device__ void do_poison(int* poison) { *(volatile int*)poison = 1; }

// stage markers for hang forensics: each kernel writes stage n into flags[288+n]
// (plain store, visible to host read_words after the fact).
__device__ __forceinline__ void stage(unsigned* my_flags, int n) {
  store_plain(&my_flags[288 + n], 0x57450000u + (unsigned)n);  // "WE" + stage;
  // line 1152+ — separate from F_SEQ (1024) and F_SEEN (1088) lines.
}

// Device rendezvous (block 0 only): 64-slot ring, exact-match, plain stores,
// reposting. Absorbs kernel-launch/queue skew before the timed data spin starts.
// Values are the SPMD seq itself (monotone, both ranks identical) — no magic
// constants in the seq space; exact equality on a ring slot; one line per writer.
#define F_HS_A(s) (32 * 16 + 16 * ((s) & 63u))    // lines 32..95
#define F_HS_B(s) (96 * 16 + 16 * ((s) & 63u))    // lines 96..159
__device__ bool rendezvous_blk0(unsigned* peer_flags, unsigned* my_flags, int role,
                                unsigned s) {
  const unsigned long long rcap = SPIN_CAP;
  const unsigned i_a = F_HS_A(s), i_b = F_HS_B(s);
  unsigned long long n = 0;
  if (role == 0) {
    store_plain(peer_flags + i_a, s);          // heartbeat
    while (load_plain(my_flags + i_b) != s) {  // wait echo
      if (++n > rcap) return false;
      if ((n & 255u) == 0u) store_plain(peer_flags + i_a, s);
      bspin_backoff();
    }
  } else {
    while (load_plain(my_flags + i_a) != s) {  // wait heartbeat
      if (++n > rcap) return false;
      if ((n & 255u) == 0u) store_plain(peer_flags + i_b, s);
      bspin_backoff();
    }
    store_plain(peer_flags + i_b, s);          // echo
  }
  return true;
}

// bf16x8 add of two int4s in fp32 (matches V0 numerics: RCCL sums bf16 via fp32
// accumulation? V0 used fp32 accumulate too — byte-identity target is the same).
__device__ __forceinline__ int4 vadd_bf16(const int4& a, const int4& c) {
  int4 r;
  const __hip_bfloat16* pa = (const __hip_bfloat16*)&a;
  const __hip_bfloat16* pc = (const __hip_bfloat16*)&c;
  __hip_bfloat16* pr = (__hip_bfloat16*)&r;
  #pragma unroll
  for (int j = 0; j < 8; ++j)
    pr[j] = (__hip_bfloat16)(__bfloat162float(pa[j]) + __bfloat162float(pc[j]));
  return r;
}

// ---- F: flush variant (single block). in/out: bf16, n16 16B words. ----
__global__ void ar_f(const int4* __restrict__ in4, int4* peer_base, const int4* my_base,
                     int4* __restrict__ out4,
                     unsigned* peer_flags, unsigned* my_flags, unsigned* seq_ctr,
                     unsigned* hz_ctr, int* poison, unsigned long long* spin_stats,
                     int n16, long long slot_stride16, int use_hz, int role) {
  const int tid = threadIdx.x, nt = blockDim.x;
  __shared__ unsigned s_seq;
  __shared__ unsigned s_ok;
  if (tid == 0) {
    s_seq = atomicAdd(seq_ctr, 1u) + 1u;
    s_ok = use_hz ? (rendezvous_blk0(peer_flags, my_flags, role, s_seq) ? 1u : 0u) : 1u;
    if (!s_ok) do_poison(poison);
  }
  __syncthreads();
  if (!s_ok) return;
  const unsigned s = s_seq;
  const int slot = (int)(s & 1u);
  int4* ps = peer_base + (long long)slot * slot_stride16;
  const int4* ms = my_base + (long long)slot * slot_stride16;

  for (int k = tid; k < n16; k += nt) ps[k] = in4[k];   // push (plain, vectorized)
  asm volatile("s_waitcnt vmcnt(0) expcnt(0)" ::: "memory");
  __syncthreads();
  if (tid == 0) {
    stage(my_flags, 1);                                  // push complete
    store_plain(&peer_flags[F_SEQ], s);                  // flag (plain + flush)
    stage(my_flags, 2);                                  // flag posted
    unsigned long long spins = 0;
    bool ok = false;
    const int spin_mode = use_hz >> 1;                   // bit1: 0=atomic-acq 1=plain-ld
    while (!ok) {
      unsigned v = spin_mode ? load_plain(&my_flags[F_SEQ])
                             : load_acq(&my_flags[F_SEQ]);
      if (v >= s) ok = true;
      else if (++spins > SPIN_CAP) break;
      else bspin_backoff();
    }
    *spin_stats = spins;
    store_plain(&my_flags[F_SEEN], load_plain(&my_flags[F_SEQ]));  // what my line holds
    s_ok = ok ? 1u : 0u;
    if (!ok) { stage(my_flags, 3); do_poison(poison); }
  }
  __syncthreads();
  if (!s_ok) return;                                     // NEVER reduce stale data
  for (int k = tid; k < n16; k += nt) out4[k] = vadd_bf16(in4[k], ms[k]);
}

// ---- L: LL variant (single block). 8B words: {payload u32 (2 bf16), seq u32}. ----
// Scratch slots: 2 x slot_stride8 u64 words. No flag line at all.
__global__ void ar_l(const unsigned* __restrict__ in2,
                     unsigned long long* peer_base, const unsigned long long* my_base,
                     unsigned* __restrict__ out2,
                     unsigned* seq_ctr, unsigned* hz_ctr,
                     unsigned* peer_flags, unsigned* my_flags,  // hz only
                     int* poison, unsigned long long* spin_stats,
                     int n8, long long slot_stride8, int use_hz, int role) {
  const int tid = threadIdx.x, nt = blockDim.x;
  __shared__ unsigned s_seq;
  __shared__ unsigned s_ok;
  if (tid == 0) {
    s_seq = atomicAdd(seq_ctr, 1u) + 1u;
    s_ok = use_hz ? (rendezvous_blk0(peer_flags, my_flags, role, s_seq) ? 1u : 0u) : 1u;
    if (!s_ok) do_poison(poison);
  }
  __syncthreads();
  if (!s_ok) return;
  const unsigned s = s_seq;
  const int slot = (int)(s & 1u);
  unsigned long long* ps = peer_base + (long long)slot * slot_stride8;
  const unsigned long long* ms = my_base + (long long)slot * slot_stride8;

  for (int k = tid; k < n8; k += nt)
    ps[k] = ((unsigned long long)s << 32) | in2[k];      // 8B payload+tag
  asm volatile("s_waitcnt vmcnt(0) expcnt(0)" ::: "memory");
  __syncthreads();
  // consume: each thread validates the tag of every word it reduces
  unsigned long long max_spins = 0;
  for (int k = tid; k < n8; k += nt) {
    unsigned long long spins = 0;
    bool ok = false;
    unsigned v = 0;
    while (!ok) {
      unsigned long long w64 = load_acq64(&ms[k]);
      unsigned tag = (unsigned)(w64 >> 32);
      if (tag == s) { v = (unsigned)w64; ok = true; }
      else if (tag > s || ++spins > SPIN_CAP) break;     // impossible tag or timeout
      else bspin_backoff();
    }
    if (spins > max_spins) max_spins = spins;
    if (!ok) { s_ok = 0; do_poison(poison); break; }
    unsigned mine = in2[k];
    __hip_bfloat16 m0 = (__hip_bfloat16)(unsigned)(mine & 0xFFFFu);
    __hip_bfloat16 m1 = (__hip_bfloat16)(unsigned)(mine >> 16);
    __hip_bfloat16 t0 = (__hip_bfloat16)(unsigned)(v & 0xFFFFu);
    __hip_bfloat16 t1 = (__hip_bfloat16)(unsigned)(v >> 16);
    __hip_bfloat16 r0 = (__hip_bfloat16)(__bfloat162float(m0) + __bfloat162float(t0));
    __hip_bfloat16 r1 = (__hip_bfloat16)(__bfloat162float(m1) + __bfloat162float(t1));
    out2[k] = ((unsigned)*(unsigned short*)&r1 << 16) | *(unsigned short*)&r0;
  }
  if (tid == 0) *spin_stats = max_spins;
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
static std::tuple<int64_t, int64_t> alloc_host_mapped() {
  void* h = nullptr;
  HIPOK(hipHostMalloc(&h, 64, hipHostMallocMapped));
  std::memset(h, 0, 64);
  void* d = nullptr;
  HIPOK(hipHostGetDevicePointer(&d, h, 0));
  return std::make_tuple((int64_t)h, (int64_t)d);
}

static void ar_f_run(int64_t peer_scratch, int64_t my_scratch, int64_t peer_flags,
                     int64_t my_flags, int64_t seq_ctr, int64_t hz_ctr,
                     int64_t poison_dev, int64_t spin_stats,
                     int64_t inp, int64_t out, int64_t n_elem,
                     int64_t slot_stride16, int64_t nthreads,
                     int64_t use_hz, int64_t role, int64_t stream_i) {
  const int n16 = (int)(n_elem * 2 / 16);
  ar_f<<<1, (int)nthreads, 0, (hipStream_t)stream_i>>>(
      (const int4*)inp, (int4*)peer_scratch, (const int4*)my_scratch, (int4*)out,
      (unsigned*)peer_flags, (unsigned*)my_flags, (unsigned*)seq_ctr,
      (unsigned*)hz_ctr, (int*)poison_dev, (unsigned long long*)spin_stats,
      n16, (long long)slot_stride16, (int)use_hz, (int)role);
  HIPOK(hipGetLastError());
}
static void ar_l_run(int64_t peer_scratch, int64_t my_scratch, int64_t peer_flags,
                     int64_t my_flags, int64_t seq_ctr, int64_t hz_ctr,
                     int64_t poison_dev, int64_t spin_stats,
                     int64_t inp, int64_t out, int64_t n_elem,
                     int64_t slot_stride8, int64_t nthreads,
                     int64_t use_hz, int64_t role, int64_t stream_i) {
  const int n8 = (int)(n_elem * 2 / 4);
  ar_l<<<1, (int)nthreads, 0, (hipStream_t)stream_i>>>(
      (const unsigned*)inp, (unsigned long long*)peer_scratch,
      (const unsigned long long*)my_scratch, (unsigned*)out,
      (unsigned*)seq_ctr, (unsigned*)hz_ctr,
      (unsigned*)peer_flags, (unsigned*)my_flags,
      (int*)poison_dev, (unsigned long long*)spin_stats,
      n8, (long long)slot_stride8, (int)use_hz, (int)role);
  HIPOK(hipGetLastError());
}

PYBIND11_MODULE(ar_proto, m) {
  m.def("alloc_shared", &alloc_shared, py::arg("size"), py::arg("finegrained") = true);
  m.def("open_shared", &open_shared);
  m.def("memzero", &memzero);
  m.def("read_words", &read_words);
  m.def("alloc_host_mapped", &alloc_host_mapped);
  m.def("ar_f", &ar_f_run);
  m.def("ar_l", &ar_l_run);
  m.attr("SPIN_CAP") = SPIN_CAP;
  m.attr("BACKOFF") = BACKOFF;
}
