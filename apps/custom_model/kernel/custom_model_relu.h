#ifndef _CUSTOM_MODEL_MAT_VEC_MUL_H_
#define _CUSTOM_MODEL_MAT_VEC_MUL_H_

#include <stdint.h>
#include <riscv_vector.h>

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


#endif