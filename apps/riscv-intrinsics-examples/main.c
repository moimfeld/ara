// Author: Moritz Imfeld <moimfeld@student.ethz.ch>

// How to compile:
// ENV_DEFINES="-DUSE_VEXT=1 -DDEBUG=1 -DFC_MODEL" make bin/custom_model
// ENV_DEFINES="-DUSE_VEXT=1 -DDEBUG=1 -DFC_MODEL" make bin/custom_model.spike # for spike
//
// -D DEBUG enables debug prints
// -D FC_MODEL selects the model that should be evaluated/run
// -D USE_VECT enables vectorized kernels
//
// To compile for kernel power analysis
// ENV_DEFINES="-DVCD_DUMP -DUSE_VEXT=1 -DDEBUG=1 -DFC_MODEL -DPA_<kernel>" make bin/custom_model
//
// example command for power analysis script
// /scratch/vlsi4_07/power_analysis/scripts/ara_power_wrap.sh 1 4 custom_model "-DUSE_VEXT=1 -DPA_RELU=1" 0 128
//

#include "kernel/rvv_reduce.h"
#include <stdint.h>
#include <string.h>

#include "util.h"
#include "runtime.h"

#ifndef SPIKE
#include "printf.h"
#else
#include "util.h"
#include <stdio.h>
#endif

int main() {

  // POWER ANALYSIS NOT YET IMPLEMENTED
  // Call power_analysis() function if VCD_DUMP is enabled
  // #if defined(VCD_DUMP)
  // power_analysis();
  // return 0;

  // MEASURING NOT YET IMPLEMENTED
  // Run forward-pass if VCD_DUMP is not defined
  // #if defined(MEASURE)
  // printf("Measuring enabled\n");
  // #endif

  // reduction kernel
  // reduce_main();

    const int N = 31;
  // uint32_t seed = 0xdeadbeef;
  // srand(seed);

  // data gen
  printf("Starting initializing Arrays\n");
  double A[N], B[N];
  // gen_rand_1d(A, N);
  // gen_rand_1d(B, N);
  for (int i = 0; i < N; i++) {
    A[i] = i * 1.0f + 20.0f;
    B[i] = i * 1.0f + 10.0f;
  }
  printf("Finished initializing Arrays\n");


  // compute
  double golden_sum, actual_sum;
  int golden_count, actual_count;
  printf("Starting golden reduction\n");
  start_timer();
  reduce_golden(A, B, &golden_sum, &golden_count, N);
  stop_timer();
  int64_t reduce_golden_time = get_timer();
  printf("MEASUREMENT: reduce_golden execution took %0ld cycles\n", reduce_golden_time);
  printf("Finished golden reduction\n");
  printf("Starting vector reduction\n");
  start_timer();
  reduce_vec(A, B, &actual_sum, &actual_count, N);
  stop_timer();
  int64_t reduce_vec_time = get_timer();
  printf("MEASUREMENT: reduce_vec execution took %0ld cycles\n", reduce_vec_time);
  printf("Finished vector reduction\n");

  // compare
  if (golden_count == actual_count) { // ADD: golden_sum - actual_sum < 1e-6) && 
    printf("reduction pass\n", golden_count, actual_count);
  } else {
    // double difference = golden_sum - actual_sum;
    printf("reduction fail (golden_sum = %x, actual_sum = %x, golden_count = %0d, actual_count = %0d)\n", golden_sum, actual_sum, golden_count, actual_count);
  }

  return 0;
}