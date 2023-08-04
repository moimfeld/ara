// Author: Moritz Imfeld <moimfeld@student.ethz.ch>

#include <stdint.h>
#include <string.h>
#include <riscv_vector.h>

#ifndef SPIKE
#include "printf.h"
#else
#include "util.h"
#include <stdio.h>
#endif

int main() {

  // output array
  float output[10];

  // init variables
  volatile float zio = 0.1337F;
  asm volatile("vsetivli zero, 16, e32, m8, ta, ma");
  asm volatile("vfmv.v.f v0, %0" :: "f"(zio));

  #if defined(STORE)
  // Adjust vl for bias addition
  asm volatile("vsetivli zero, 1, e32, m1, ta, ma");
  // Store result
  asm volatile("vse32.v v0, (%0)"::"r"(&output[0]));
  #else
  asm volatile("vfmv.f.s %0, v0" : "=f"(output[0]));
  #endif

  // print output[0]
  printf("output[0] = %f\n", output[0]);

  return 0;
}
