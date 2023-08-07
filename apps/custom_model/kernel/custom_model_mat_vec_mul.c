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
        res_vec = vfsub_vv_f32m1(res_vec, res_vec, vl_max);
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

// Written using inline assembly
// vector regirster usage in function:
// v0: result vector (m8)
// v8: input (m8)
// v16: weights, multiplication results (m8)
void mat_vec_mul_n_columns_smaller_vl_max(const uint32_t n_rows,
                                          const uint32_t n_columns,
                                          const float   weights[n_rows][n_columns],
                                          const float   *bias,
                                          float *input,
                                          float *output)
{
    // Initialize
    float * w_ptr = (float * ) weights;
    size_t vl;
    asm volatile("vsetvli %0, %1, e32, m8, ta, ma" : "=r"(vl) :"r"(n_columns));

    // loading the input vector only once
    asm volatile("vle32.v v8, (%0)" ::"r"(input)); // load input once
    // inp_vec = vle32_v_f32m8(input, vl);

    for (uint32_t i = 0; i < n_rows; i++) {
        // FOR TESTING
        asm volatile("vsetvli %0, %1, e32, m8, ta, ma" : "=r"(vl) :"r"(n_columns));

        DEBUG_PRINTF("mat_vec_mul_n_columns_smaller_vl_max: Starting to process %0dth row\n", i)
         // load weights
        asm volatile("vle32.v v16, (%0)" :: "r"(w_ptr));

         // init result vector to 0
        asm volatile("vmv.s.x v0, x0");

        // computation
        asm volatile("vfmul.vv v16, v16, v8");

        // Updating pointers
        w_ptr += vl;

        // Sum up multiplications
        asm volatile("vfredusum.vs v0, v16, v0");

        // store result in output buffer
        asm volatile("vfmv.f.s %0, v0" : "=f"(output[i])); // USABLE NOW (on verilator) :)
        
        // Adjust vl for bias addition
        // asm volatile("vsetivli zero, 1, e32, m1, ta, ma");

        // Store result
        // asm volatile("vse32.v v0, (%0)"::"r"(&output[i]));

        // Add bias
        output[i] += bias[i];
    }
    return;
}
