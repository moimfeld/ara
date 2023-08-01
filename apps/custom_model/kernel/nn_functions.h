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

