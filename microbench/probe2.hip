// probe2.hip — isolate why s_timerhi compiles in probe_isa but not in ar_phase0.
#include <hip/hip_runtime.h>
#include <cstdint>
__device__ unsigned long long gt_a() {  // probe pattern: direct pointer store
  uint32_t lo, hi0, hi1;
  do {
    asm volatile("s_timerhi %0" : "=s"(hi0));
    asm volatile("s_timerlo %0" : "=s"(lo));
    asm volatile("s_timerhi %0" : "=s"(hi1));
  } while (hi0 != hi1);
  return ((unsigned long long)hi0 << 32) | lo;
}
__device__ unsigned long long gt_b() {  // + memory clobber
  uint32_t lo, hi0, hi1;
  do {
    asm volatile("s_timerhi %0" : "=s"(hi0) :: "memory");
    asm volatile("s_timerlo %0" : "=s"(lo) :: "memory");
    asm volatile("s_timerhi %0" : "=s"(hi1) :: "memory");
  } while (hi0 != hi1);
  return ((unsigned long long)hi0 << 32) | lo;
}
__device__ unsigned long long gt_c() {  // 1 MHz coarse: lo only
  uint32_t lo;
  asm volatile("s_timerlo %0" : "=s"(lo) :: "memory");
  return lo;
}
__device__ unsigned long long gt_d() { return clock64(); }
__global__ void k(unsigned long long* o) { o[0] = gt_a(); o[1] = gt_b(); o[2] = gt_c(); o[3] = gt_d(); }
