#include "forward_pass.h"
#include "nn_functions.h"
#include "custom_model_macros.h"
#include "runtime.h"
#include <stdint.h>
#include "util.h"

#if defined(USE_VEXT)
#include <riscv_vector.h>
#endif

#ifndef SPIKE
#include "printf.h"
#else
#include <stdio.h>
#endif

int8_t float_mat_vec_product(const uint32_t n_rows,
                             const uint32_t n_columns,
                             const float   weights[n_rows][n_columns],
                             const float   *bias,
                             float *input,
                             float *output)
{
    // VECTORIZED CODE
    #if defined(USE_VEXT)
    size_t vl_max = vsetvl_e32m8(n_columns);
    DEBUG_PRINTF("vl_max = %0d\n", vl_max);
    if (vl_max < n_columns) {
        slow_mat_vec_mul(n_rows, n_columns, weights, bias, input, output);
    } else if (vl_max == n_columns) {
        mat_vec_mul_n_columns_smaller_vl_max(n_rows, n_columns, weights, bias, input, output);
    }

    // NON VECTORIZED CODE
    #else
    for (uint32_t i = 0; i < n_rows; i++) {
        for (uint32_t j = 0; j < n_columns; j++) {
            output[i] += weights[i][j] * input[j];
        }
        output[i] += bias[i];
    }
    #endif

    return 0;
}

int8_t conv_2d(const uint32_t img_n_rows,
               const uint32_t img_n_columns,
               const uint32_t filter_size,
               const float    filter[filter_size][filter_size],
               const float    *bias,
               float          *input,
               float          *output)
{
    // VECTORIZED CODE
    // #if defined(USE_VEXT)
    // #else
    for (uint32_t img_row = 0; img_row < img_n_rows - 2; img_row++) {
        for (uint32_t img_column = 0; img_column < img_n_columns - 2; img_column++) {
            int32_t output_offset = img_row * (img_n_rows - 2) + img_column;
            output[output_offset] = bias[0];
            for (uint32_t f_row = 0; f_row < filter_size; f_row++) {
                for (uint32_t f_column = 0; f_column < filter_size; f_column++) {
                    output[output_offset] += input[output_offset + f_row * img_n_rows + f_column] * filter[f_row][f_column];
                }
            }
        }
    }
    return 0;
    // #endif
}


void slow_mat_vec_mul(const uint32_t n_rows,
                        const uint32_t n_columns,
                        const float   weights[n_rows][n_columns],
                        const float   *bias,
                        float *input,
                        float *output)
{
    // Initialize
    float * w_ptr = (float * ) weights;
    size_t vl_max = vsetvl_e32m8(n_columns);
    vfloat32m8_t acc_vec;
    vfloat32m1_t res_vec;

    for (uint32_t i = 0; i < n_rows; i++) {
        // Initialization
        uint32_t remaining_columns = n_columns;
        acc_vec = vfsub_vv_f32m8(acc_vec, acc_vec, vl_max);
        res_vec = vfsub_vv_f32m1(res_vec, res_vec, 1);
        float * i_ptr = input;

        // Vectorized Multiplication loop
        while(remaining_columns > 0) {
            // Update vl
            size_t vl = vsetvl_e32m8(remaining_columns);

            // computation
            acc_vec = vfmacc_vv_f32m8(acc_vec, vle32_v_f32m8(w_ptr, vl), vle32_v_f32m8(i_ptr, vl), vl);

            // Updating pointers
            w_ptr += vl;
            i_ptr += vl;
            remaining_columns -= vl;
        }
        // Accumulate accumulators (reduction)
        res_vec = vfredosum_vs_f32m8_f32m1(res_vec, acc_vec, res_vec, n_columns);
        
        // Adjust vl for bias addition
        size_t vl = vsetvl_e32m1(1);

        // Add bias
        res_vec = vfadd_vf_f32m1(res_vec, bias[i], 1);

        // Store result
        vse32_v_f32m1(&output[i], res_vec, 1);

        // #pragma clang optimize off
        // float cringe = (float) vfmv_f_s_f32m1_f32(res_vec);
        // printf("output[%0d] = %f\n", (cringe + bias[i])); // vfmv_f_s_f32m1_f32 not working for some reason
        // #pragma clang optimize on
    }
}

void mat_vec_mul_n_columns_smaller_vl_max(const uint32_t n_rows,
                                          const uint32_t n_columns,
                                          const float   weights[n_rows][n_columns],
                                          const float   *bias,
                                          float *input,
                                          float *output)
{
    // Initialize
    float * w_ptr = (float * ) weights;
    size_t vl = vsetvl_e32m8(n_columns);
    vfloat32m8_t acc_vec;
    vfloat32m8_t inp_vec;
    vfloat32m1_t res_vec;

    // loading the input vector only once
    inp_vec = vle32_v_f32m8(input, vl);

    for (uint32_t i = 0; i < n_rows; i++) {
        DEBUG_PRINTF("Starting to process %0dth row\n", i)
        // Initialization
        vsetvl_e32m8(n_columns);
        acc_vec = vfsub_vv_f32m8(acc_vec, acc_vec, vl);
        res_vec = vfsub_vv_f32m1(res_vec, res_vec, 1);

        // computation
        acc_vec = vfmacc_vv_f32m8(acc_vec, vle32_v_f32m8(w_ptr, vl), inp_vec, vl);

        // Updating pointers
        w_ptr += vl;

        // Accumulate accumulators (reduction)
        res_vec = vfredosum_vs_f32m8_f32m1(res_vec, acc_vec, res_vec, n_columns);
        
        // Adjust vl for bias addition
        // size_t vl = vsetvl_e32m1(1);

        // Add bias
        // res_vec = vfadd_vf_f32m1(res_vec, bias[i], 1);

        // Store result
        // vse32_v_f32m1(&output[i], res_vec, 1);

        // WHY DOES THIS LINE NOT REPLACE THE THREE VECTOR INSTRUCTIONS ABOVE?????? (IN SPIKE IT WORKS)
        float a = vfmv_f_s_f32m1_f32(res_vec);
        output[i] = a + bias[i];
        printf("output[%0d] = %f", i, a);
    }
    return;
}
