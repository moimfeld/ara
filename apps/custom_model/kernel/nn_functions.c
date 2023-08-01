#include "custom_model_forward_pass.h"
#include "custom_model_mat_vec_mul.h"
#include "custom_model_conv2d.h"
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
    DEBUG_PRINTF("float_mat_vec_product: vl_max = %0d\n", vl_max);
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

// MOIMFELD: WARNING, at the moment it is only working in vectorized manner in 3x3
int8_t conv_2d(const uint32_t img_n_rows,
               const uint32_t img_n_columns,
               const uint32_t filter_size,
               const float    filter[filter_size][filter_size],
               const float    *bias,
               float          *input,
               float          *output)
{
    // VECTORIZED CODE
    #if defined(USE_VEXT)
    fconv2d_tensor32_vec_6xC_3x3(output, input, filter, bias[0], img_n_rows, img_n_columns, 1, filter_size); // only one input channel at the moment --> else change second last argument and pass bias accordingly
    #else
    // reference kernel
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
    #endif
}


