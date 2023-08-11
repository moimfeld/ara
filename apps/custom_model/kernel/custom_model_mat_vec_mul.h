#ifndef _CUSTOM_MODEL_MAT_VEC_MUL_H_
#define _CUSTOM_MODEL_MAT_VEC_MUL_H_

#include <stdint.h>

extern int64_t event_trigger;


// First version of matrix-vector multiplication
//
// Outer-loop:  Initilization and reduction
//              1. Zeroing accumulate vector and result vector
//              2. Calling innter loop
//              3. Reducing accumulated results to a scalar
//              4. Add bias to scalar
//              5. save scalar in output vector
//
// Inner-loop:  Stripmining to get elementwise MACC of weights and input
//              1. Load weights and input
//              2. MACC

void slow_mat_vec_mul(const uint32_t n_rows,
                      const uint32_t n_columns,
                      const float   weights[n_rows][n_columns],
                      const float   *bias,
                      float *input,
                      float *output);

// Fast matrix-vector multiplication for "small" inputs/matrices
//
// Init:        1. Load input once
// 
// Outer-loop:  Initilization and reduction
//              1. Zeroing accumulate vector and result vector
//              2. MUL row * inputs
//              3. Reducing accumulated results to a scalar
//              4. Add bias to scalar
//              5. save scalar in output vector

void mat_vec_mul_n_columns_smaller_vl_max(const uint32_t n_rows,
                                          const uint32_t n_columns,
                                          const float   weights[n_rows][n_columns],
                                          const float   *bias,
                                          float *input,
                                          float *output);

#endif