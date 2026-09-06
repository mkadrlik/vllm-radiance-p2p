// probe3.hip — does ANY ns clock work in a real gfx1100 kernel?
#include <hip/hip_runtime.h>
#include <cstdint>
__global__ void k3a(unsigned long long* o) { *o = 0ull; }
__global__ void k3b(unsigned long long* o) { unsigned long long t; asm volatile("s_memrealtime %0" : "=s"(t)); *o = t; }
__global__ void k3c(unsigned long long* o) {  // host-side event timing of a spin kernel
  unsigned long long s = 0;
  for (int i = 0; i < 100000; ++i) s += i;
  *o = s;
}
