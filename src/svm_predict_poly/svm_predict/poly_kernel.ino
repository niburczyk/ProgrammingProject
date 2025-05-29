#ifndef VEC_DIM
#error "VEC_DIM ist nicht definiert"
#endif

#ifndef GAMMA
#error "GAMMA ist nicht definiert"
#endif

#ifndef COEF0
#error "COEF0 ist nicht definiert"
#endif

#ifndef DEGREE
#error "DEGREE ist nicht definiert"
#endif

#include <avr/pgmspace.h>
#include <math.h>

inline float polynomial_kernel(const float* u_progmem, const float* v) {
  float result = 0.0;
  for (int i = 0; i < VEC_DIM; ++i) {
    result += pgm_read_float(u_progmem + i) * v[i];
  }
  return powf((GAMMA * result + COEF0), DEGREE);
}
