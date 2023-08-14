#include "custom_model_padding.h"
#include <riscv_vector.h>
#ifndef SPIKE
#include "printf.h"
#else
#include <stdio.h>
#endif


// STRIPMINING IS NOT TESTED!!!!!!!! (works 100% if i_columns < (vl_max - 2*pad))
void float_zero_pad(float *output, float *input, uint64_t i_rows, uint64_t i_columns, uint64_t pad){
    #if defined(USE_VEXT)
    uint32_t o_rows = i_rows + 2*pad;

    vfloat32m8_t res_vec;

    // start VCD_DUMP
    #if defined(VCD_DUMP)
    event_trigger = +1;
    #endif

    // top row padding
    for (uint32_t i = 0; i < pad; i++) {
        // implement strip mining for large images
        uint64_t remaining_col = o_rows;
        while(remaining_col > 0) {
            // set to max possible columns
            size_t cols = vsetvl_e32m8(remaining_col);

            // zero vector and store in output array
            res_vec = vfsub_vv_f32m8(res_vec, res_vec, cols);
            vse32_v_f32m8(output, res_vec, cols);

            // bump pointer and 
            remaining_col -= cols;
            output += cols;
        }
    }

    // left and right padding
    for (uint32_t i = 0; i < i_rows; i++) {
        uint32_t padding_itr = 0;
        uint32_t remaining_col = i_rows + pad; // input rows + left padding
        size_t cols = vsetvl_e32m8(remaining_col);
        res_vec = vfsub_vv_f32m8(res_vec, res_vec, cols);
        while(remaining_col > 0) {
            // load input row and bump input pointer
            size_t i_cols = vsetvl_e32m8(i_rows); // need to keep track of this column number of to bump input pointer
            res_vec = vle32_v_f32m8(input, i_cols);

            // set len to reamining_col (potentially the same vector length)
            size_t cols = vsetvl_e32m8(remaining_col);

            // left pad vector
            if (padding_itr == 0) {
                // left padding
                for (uint32_t i = 0; i < pad; i++) {
                    output[0] = 0;
                    output += 1;
                }
            }
            vse32_v_f32m8(output, res_vec, cols);

            // bump pointer and 
            remaining_col -= cols;
            output += cols;
            input  += i_rows;
            padding_itr += 1;
        }
    }

    // bottom padding
    for (uint32_t i = 0; i < pad; i++) {
        // implement strip mining for large images
        uint64_t remaining_col = o_rows;
        while(remaining_col > 0) {
            // set to max possible columns
            size_t cols = vsetvl_e32m8(remaining_col);

            // zero vector and store in output array
            res_vec = vfsub_vv_f32m8(res_vec, res_vec, cols);
            vse32_v_f32m8(output, res_vec, cols);

            // bump pointer and 
            remaining_col -= cols;
            output += cols;
        }
    }
    // stop VCD_DUMP
    #if defined(VCD_DUMP)
    event_trigger = -1;
    #endif
    #else
    for (uint32_t i = 0; i < i_rows + 2*pad; i++) {
        for (uint32_t j = 0; j < i_columns + 2*pad; j++) {
            if (i == 0 || i == i_rows-1 || j == 0 || j == i_columns-1) {
                output[i * (i_rows + 2*pad) + j] = 0;
            } else {
                output[i * (i_rows + 2*pad) + j] = input[(i-1) * (i_rows) + (j-1)];
            }
        }
    }
    #endif
}
