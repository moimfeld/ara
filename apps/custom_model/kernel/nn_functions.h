#ifndef _NN_FUNCTIONS_H_
#define _NN_FUNCTIONS_H_

#include <stdint.h>

int8_t float_mat_vec_product(const uint32_t n_rows,
                             const uint32_t n_columns,
                             const float   weights[n_rows][n_columns],
                             const float   *bias,
                             float *input,
                             float *output);

int8_t conv_2d(const uint32_t img_n_rows,
               const uint32_t img_n_columns,
               const uint32_t filter_size,
               const float    filter[filter_size][filter_size],
               const float    *bias,
               float          *input,
               float          *output);

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

// Simulation results for this algorithm
// Simulation running, end by pressing CTRL-c.
// Copied input
// The execution took 442688 cycles. # first layer
// The execution took 19518 cycles. # second layer
// The execution took 5367 cycles. # third layer
// The execution took 1999 cycles. # output layer
// Argmax = 7
// Expected = 7
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

// Simulation results for this algorithm
// Simulation running, end by pressing CTRL-c.
// Copied input
// dense_0_weight_columns = 784
// The execution took 431757 cycles.
// First layer done, very cool :)
// The execution took 18926 cycles.
// Second layer done, very cool :)
// The execution took 5132 cycles.
// Third layer done, very cool :)
// The execution took 1865 cycles.
// Last layer done, very cool :)
// Argmax = 7
// Expected = 7
void mat_vec_mul_n_columns_smaller_vl_max(const uint32_t n_rows,
                                          const uint32_t n_columns,
                                          const float   weights[n_rows][n_columns],
                                          const float   *bias,
                                          float *input,
                                          float *output);


// Inline functions
__attribute__((always_inline)) static int8_t float_relu(const uint32_t len,
                                                        float         *input,
                                                        float         *output)
{
    #if defined(USE_VEXT)
    // Initialize
    uint32_t remaining_columns = len;
    vfloat32m8_t out_vec;

    // Vectorized vmax loop
    while (remaining_columns > 0) {
        // Update vl
        size_t vl = vsetvl_e32m8(remaining_columns);

        // Computation
        out_vec = vfmax_vf_f32m8(vle32_v_f32m8(input, vl), 0.0, vl);

        // Store result
        vse32_v_f32m8(output, out_vec, vl);

        // Update pointers
        remaining_columns -=vl;
        input  += vl;
        output += vl;
    }
    #else
    for (uint32_t i = 0; i < len; i++) {
        output[i] = (input[i] > 0.0) ? input[i] : 0.0;
    }
    #endif
    return 0;
}


__attribute__((always_inline)) static uint32_t float_argmax(const uint32_t len,
                                           float *input)
{
    float max;
    uint32_t argmax;
    for(uint32_t i = 0; i < len; i++) {
        if (i == 0) {
            max = input[i];
            argmax = i;
        } else {
            if (max < input[i]) {
            max = input[i];
            argmax = i;
            }
        }
    }
    return argmax;
}

#endif

