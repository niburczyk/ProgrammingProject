#ifndef VEC_DIM
#error "VEC_DIM ist nicht definiert"
#endif

#ifndef GAMMA
#error "GAMMA ist nicht definiert"
#endif

#include <avr/pgmspace.h>
#include <math.h>

// RBF-Kernel: exp(-gamma * ||u - v||^2)
inline float rbf_kernel(const float* u_progmem, const float* v) {
  float dist2 = 0.0;
  for (int i = 0; i < VEC_DIM; ++i) {
    float u_i = pgm_read_float(u_progmem + i);
    float diff = u_i - v[i];
    dist2 += diff * diff;
  }
  return expf(-GAMMA * dist2);
}
