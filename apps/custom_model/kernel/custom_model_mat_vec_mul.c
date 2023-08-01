#include <riscv_vector.h>
#include "custom_model_mat_vec_mul.h"
#include "custom_model_macros.h"
#include "stdio.h"

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
        DEBUG_PRINTF("mat_vec_mul_n_columns_smaller_vl_max: Starting to process %0dth row\n", i)
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
        size_t vl = vsetvl_e32m1(1);

        // Add bias
        res_vec = vfadd_vf_f32m1(res_vec, bias[i], 1);

        // Store result
        vse32_v_f32m1(&output[i], res_vec, 1);

        // WHY DOES THIS LINE NOT REPLACE THE THREE VECTOR INSTRUCTIONS ABOVE?????? (IN SPIKE IT WORKS)
        // output_[i] = vfmv_f_s_f32m1_f32(res_vec) + bias[i];
        // printf("output[%0d] = %f", i, a);
    }
    return;
}
