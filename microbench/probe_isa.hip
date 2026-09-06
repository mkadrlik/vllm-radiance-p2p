// probe_isa.hip — find working gfx1100 timing + L2-flush spellings.
#include <hip/hip_runtime.h>
#include <cstdint>
__device__ void t1(uint32_t* o) { asm volatile("s_timerlo %0" : "=s"(*o)); }
__device__ void t2(uint32_t* o) { asm volatile("s_timerhi %0" : "=s"(*o)); }
__device__ void t3(uint64_t* o) {
  asm volatile("s_timerinit");
  asm volatile("s_timer %0, %1" : "=s"(((uint32_t*)o)[0]), "=s"(((uint32_t*)o)[1]));
}
__device__ void t4(uint64_t* o) { *o = __builtin_amdgcn_wallclock(); }
__device__ void t5(uint64_t* o) { *o = clock64(); }
__device__ void t6(uint32_t* o) { *o = __builtin_amdgcn_clock(); }
__device__ void w1() { asm volatile("buffer_load_scb_m32 off, off, off, off, 0, sc0, ss0" ::: "memory"); }
__device__ void w2() { asm volatile("buffer_inv sc0 ss0" ::: "memory"); }
__device__ void w3() { asm volatile("global_inv sc1 ss0" ::: "memory"); }
__device__ void w4() { asm volatile("buffer_wbl2 sc0" ::: "memory"); }
