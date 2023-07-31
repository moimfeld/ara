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
    // void fconv2d_tensor32_vec_6xC_3x3(float *o, float *i, float *f, int64_t R, int64_t C, int64_t W, int64_t F)
    // {
    #define TILE_SIZE 256
    #define TILE_SIZE_OUT TILE_SIZE - 3 + 1
    #define block_size_3x3 6
    #define next_plane_(a) ((R - a + 1)*C) << 2

    int64_t R = img_n_rows;
    int64_t C = img_n_columns;
    int64_t W = 1; // number of input channles (for now 1)
    int64_t F = filter_size;
    int64_t const ldo = (C - 2) << 2;
    int64_t const ldi = C << 2;

    int64_t vlen;

    int64_t const last_group = (R - F + 1) % block_size_3x3;

    float * f = (float *) filter;
    float * i = (float *) input;
    float * o = (float *) output;


    for (int c = 0 ; c < (C - 2) ; c += TILE_SIZE_OUT) // IF CONVOLUTION NEED TO BE TILED (C > TILE_SIZE)
    {

        float *f_ = f;
        float *i_ = i + c; 									// input pointer realtive to the tile (constant throughout the tile)
        float *o_ = o + c;									// output pointer relative to the tile		


        
        if(c > C - TILE_SIZE) 	// if we are at the right border of the input
            vlen = C % TILE_SIZE_OUT;		 	// we set the vector length to fit the last inputs
        else
            vlen = TILE_SIZE;						// else we go full length
        
        float * i__ = i_;							// inside tile pointer (change at each load)
        
        int64_t vlen_out = vlen - 2;
                


        asm volatile("vsetvli zero, %0, e32, m2, ta, ma" ::"r"(vlen));

        asm volatile("vle32.v v12, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
        asm volatile("vle32.v v14, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(next_plane_(2)));

        asm volatile("vfmul.vf v0,  v12, %0" :: "f"(f_[0]));
        asm volatile("vfmul.vf v2,  v14, %0" :: "f"(f_[0]));
        
        asm volatile("vslidedown.vi v12, v12, 1");

        asm volatile("vfmacc.vf v0,  %0, v14" ::"f"(f_[3]));

        asm volatile("vslidedown.vi v14, v14, 1");

        asm volatile("vfmacc.vf v0,  %0, v12" ::"f"(f_[1]));
        asm volatile("vfmacc.vf v2,  %0, v14" ::"f"(f_[1]));

        asm volatile("vslidedown.vi v12, v12, 1");

        asm volatile("vfmacc.vf v0,  %0, v14" ::"f"(f_[4]));

        asm volatile("vslidedown.vi v14, v14, 1");

        asm volatile("vfmacc.vf v0,  %0, v12" ::"f"(f_[2]));
        asm volatile("vfmacc.vf v2,  %0, v14" ::"f"(f_[2]));

        asm volatile("vfmacc.vf v0,  %0, v14" ::"f"(f_[5]));


        for(int ch = 1 ; ch < W ; ch ++){

            f_ += 9;

            asm volatile("vle32.v v12, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
            asm volatile("vle32.v v14, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(next_plane_(2)));

            asm volatile("vfmacc.vf v0,  %0, v12" ::"f"(f_[0]));
            asm volatile("vfmacc.vf v2,  %0, v14" ::"f"(f_[0]));

            asm volatile("vslidedown.vi v12, v12, 1");

            asm volatile("vfmacc.vf v0,  %0, v14" ::"f"(f_[3]));

            asm volatile("vslidedown.vi v14, v14, 1");
            
            asm volatile("vfmacc.vf v0,  %0, v12" ::"f"(f_[1]));
            asm volatile("vfmacc.vf v2,  %0, v14" ::"f"(f_[1]));

            asm volatile("vslidedown.vi v12, v12, 1");

            asm volatile("vfmacc.vf v0,  %0, v14" ::"f"(f_[4]));

            asm volatile("vslidedown.vi v14, v14, 1");

            asm volatile("vfmacc.vf v0,  %0, v12" ::"f"(f_[2]));
            asm volatile("vfmacc.vf v2,  %0, v14" ::"f"(f_[2]));

            asm volatile("vfmacc.vf v0,  %0, v14" ::"f"(f_[5]));

            }


        i__ = i_ + 2 * C;
        f_ = f;

        for (int r = 2 + block_size_3x3; r < R ; r += block_size_3x3)
        {

            asm volatile("vsetvli zero, %0, e32, m2, ta, ma" ::"r"(vlen));

            asm volatile("vle32.v v16, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
            asm volatile("vle32.v v18, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));

            asm volatile("vfmul.vf v4,  v16, %0" :: "f"(f_[0]));
            asm volatile("vfmacc.vf v2,  %0, v16" ::"f"(f_[3]));
            asm volatile("vfmacc.vf v0,  %0, v16" ::"f"(f_[6]));

            asm volatile("vslidedown.vi v16, v16, 1");

            asm volatile("vle32.v v20, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));

            asm volatile("vfmul.vf v6,  v18, %0" :: "f"(f_[0]));
            asm volatile("vfmacc.vf v4,  %0, v18" ::"f"(f_[3]));
            asm volatile("vfmacc.vf v2,  %0, v18" ::"f"(f_[6]));

            asm volatile("vslidedown.vi v18, v18, 1");

            asm volatile("vle32.v v22, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));

            asm volatile("vfmul.vf v8,  v20, %0" :: "f"(f_[0]));
            asm volatile("vfmacc.vf v6,  %0, v20" ::"f"(f_[3]));
            asm volatile("vfmacc.vf v4,  %0, v20" ::"f"(f_[6]));

            asm volatile("vslidedown.vi v20, v20, 1");

            asm volatile("vle32.v v24, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));

            asm volatile("vfmul.vf v10,  v22, %0" :: "f"(f_[0]));
            asm volatile("vfmacc.vf v8,  %0, v22" ::"f"(f_[3]));
            asm volatile("vfmacc.vf v6,  %0, v22" ::"f"(f_[6]));

            asm volatile("vslidedown.vi v22, v22, 1");

            asm volatile("vle32.v v26, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(next_plane_(6)));

            asm volatile("vfmul.vf v12,  v24, %0" :: "f"(f_[0]));
            asm volatile("vfmacc.vf v10,  %0, v24" ::"f"(f_[3]));
            asm volatile("vfmacc.vf v8,  %0, v24" ::"f"(f_[6]));

            asm volatile("vslidedown.vi v24, v24, 1");

            asm volatile("vfmul.vf v14,  v26, %0" :: "f"(f_[0]));
            asm volatile("vfmacc.vf v12,  %0, v26" ::"f"(f_[3]));
            asm volatile("vfmacc.vf v10,  %0, v26" ::"f"(f_[6]));

            asm volatile("vslidedown.vi v26, v26, 1");

            asm volatile("vfmacc.vf v4,  %0, v16" ::"f"(f_[1]));
            asm volatile("vfmacc.vf v2,  %0, v16" ::"f"(f_[4]));
            asm volatile("vfmacc.vf v0,  %0, v16" ::"f"(f_[7]));

            asm volatile("vslidedown.vi v16, v16, 1");

            asm volatile("vfmacc.vf v6,  %0, v18" ::"f"(f_[1]));
            asm volatile("vfmacc.vf v4,  %0, v18" ::"f"(f_[4]));
            asm volatile("vfmacc.vf v2,  %0, v18" ::"f"(f_[7]));

            asm volatile("vslidedown.vi v18, v18, 1");

            asm volatile("vfmacc.vf v8,  %0, v20" ::"f"(f_[1]));
            asm volatile("vfmacc.vf v6,  %0, v20" ::"f"(f_[4]));
            asm volatile("vfmacc.vf v4,  %0, v20" ::"f"(f_[7]));

            asm volatile("vslidedown.vi v20, v20, 1");

            asm volatile("vfmacc.vf v10,  %0, v22" ::"f"(f_[1]));
            asm volatile("vfmacc.vf v8,  %0, v22" ::"f"(f_[4]));
            asm volatile("vfmacc.vf v6,  %0, v22" ::"f"(f_[7]));

            asm volatile("vslidedown.vi v22, v22, 1");

            asm volatile("vfmacc.vf v12,  %0, v24" ::"f"(f_[1]));
            asm volatile("vfmacc.vf v10,  %0, v24" ::"f"(f_[4]));
            asm volatile("vfmacc.vf v8,  %0, v24" ::"f"(f_[7]));

            asm volatile("vslidedown.vi v24, v24, 1");

            asm volatile("vfmacc.vf v14,  %0, v26" ::"f"(f_[1]));
            asm volatile("vfmacc.vf v12,  %0, v26" ::"f"(f_[4]));
            asm volatile("vfmacc.vf v10,  %0, v26" ::"f"(f_[7]));

            asm volatile("vslidedown.vi v26, v26, 1");

            asm volatile("vfmacc.vf v4,  %0, v16" ::"f"(f_[2]));
            asm volatile("vfmacc.vf v2,  %0, v16" ::"f"(f_[5]));
            asm volatile("vfmacc.vf v0,  %0, v16" ::"f"(f_[8]));


            asm volatile("vfmacc.vf v6,  %0, v18" ::"f"(f_[2]));
            asm volatile("vfmacc.vf v4,  %0, v18" ::"f"(f_[5]));
            asm volatile("vfmacc.vf v2,  %0, v18" ::"f"(f_[8]));


            asm volatile("vfmacc.vf v8,  %0, v20" ::"f"(f_[2]));
            asm volatile("vfmacc.vf v6,  %0, v20" ::"f"(f_[5]));
            asm volatile("vfmacc.vf v4,  %0, v20" ::"f"(f_[8]));


            asm volatile("vfmacc.vf v10,  %0, v22" ::"f"(f_[2]));
            asm volatile("vfmacc.vf v8,  %0, v22" ::"f"(f_[5]));
            asm volatile("vfmacc.vf v6,  %0, v22" ::"f"(f_[8]));


            asm volatile("vfmacc.vf v12,  %0, v24" ::"f"(f_[2]));
            asm volatile("vfmacc.vf v10,  %0, v24" ::"f"(f_[5]));
            asm volatile("vfmacc.vf v8,  %0, v24" ::"f"(f_[8]));


            asm volatile("vfmacc.vf v14,  %0, v26" ::"f"(f_[2]));
            asm volatile("vfmacc.vf v12,  %0, v26" ::"f"(f_[5]));
            asm volatile("vfmacc.vf v10,  %0, v26" ::"f"(f_[8]));



            for(int ch = 1 ; ch < W ; ch ++){

                f_ += 9;

                asm volatile("vle32.v v16, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
                asm volatile("vle32.v v18, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));

                asm volatile("vfmacc.vf v4,  %0, v16" ::"f"(f_[0]));
                asm volatile("vfmacc.vf v2,  %0, v16" ::"f"(f_[3]));
                asm volatile("vfmacc.vf v0,  %0, v16" ::"f"(f_[6]));

                asm volatile("vslidedown.vi v16, v16, 1");

                asm volatile("vle32.v v20, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));

                asm volatile("vfmacc.vf v6,  %0, v18" ::"f"(f_[0]));
                asm volatile("vfmacc.vf v4,  %0, v18" ::"f"(f_[3]));
                asm volatile("vfmacc.vf v2,  %0, v18" ::"f"(f_[6]));

                asm volatile("vslidedown.vi v18, v18, 1");

                asm volatile("vle32.v v22, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));

                asm volatile("vfmacc.vf v8,  %0, v20" ::"f"(f_[0]));
                asm volatile("vfmacc.vf v6,  %0, v20" ::"f"(f_[3]));
                asm volatile("vfmacc.vf v4,  %0, v20" ::"f"(f_[6]));

                asm volatile("vslidedown.vi v20, v20, 1");

                asm volatile("vle32.v v24, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));

                asm volatile("vfmacc.vf v10,  %0, v22" ::"f"(f_[0]));
                asm volatile("vfmacc.vf v8,  %0, v22" ::"f"(f_[3]));
                asm volatile("vfmacc.vf v6,  %0, v22" ::"f"(f_[6]));

                asm volatile("vslidedown.vi v22, v22, 1");

                asm volatile("vle32.v v26, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(next_plane_(6)));

                asm volatile("vfmacc.vf v12,  %0, v24" ::"f"(f_[0]));
                asm volatile("vfmacc.vf v10,  %0, v24" ::"f"(f_[3]));
                asm volatile("vfmacc.vf v8,  %0, v24" ::"f"(f_[6]));

                asm volatile("vslidedown.vi v24, v24, 1");

                asm volatile("vfmacc.vf v14,  %0, v26" ::"f"(f_[0]));
                asm volatile("vfmacc.vf v12,  %0, v26" ::"f"(f_[3]));
                asm volatile("vfmacc.vf v10,  %0, v26" ::"f"(f_[6]));

                asm volatile("vslidedown.vi v26, v26, 1");

                asm volatile("vfmacc.vf v4,  %0, v16" ::"f"(f_[1]));
                asm volatile("vfmacc.vf v2,  %0, v16" ::"f"(f_[4]));
                asm volatile("vfmacc.vf v0,  %0, v16" ::"f"(f_[7]));

                asm volatile("vslidedown.vi v16, v16, 1");

                asm volatile("vfmacc.vf v6,  %0, v18" ::"f"(f_[1]));
                asm volatile("vfmacc.vf v4,  %0, v18" ::"f"(f_[4]));
                asm volatile("vfmacc.vf v2,  %0, v18" ::"f"(f_[7]));

                asm volatile("vslidedown.vi v18, v18, 1");

                asm volatile("vfmacc.vf v8,  %0, v20" ::"f"(f_[1]));
                asm volatile("vfmacc.vf v6,  %0, v20" ::"f"(f_[4]));
                asm volatile("vfmacc.vf v4,  %0, v20" ::"f"(f_[7]));

                asm volatile("vslidedown.vi v20, v20, 1");

                asm volatile("vfmacc.vf v10,  %0, v22" ::"f"(f_[1]));
                asm volatile("vfmacc.vf v8,  %0, v22" ::"f"(f_[4]));
                asm volatile("vfmacc.vf v6,  %0, v22" ::"f"(f_[7]));

                asm volatile("vslidedown.vi v22, v22, 1");

                asm volatile("vfmacc.vf v12,  %0, v24" ::"f"(f_[1]));
                asm volatile("vfmacc.vf v10,  %0, v24" ::"f"(f_[4]));
                asm volatile("vfmacc.vf v8,  %0, v24" ::"f"(f_[7]));

                asm volatile("vslidedown.vi v24, v24, 1");

                asm volatile("vfmacc.vf v14,  %0, v26" ::"f"(f_[1]));
                asm volatile("vfmacc.vf v12,  %0, v26" ::"f"(f_[4]));
                asm volatile("vfmacc.vf v10,  %0, v26" ::"f"(f_[7]));

                asm volatile("vslidedown.vi v26, v26, 1");

                asm volatile("vfmacc.vf v4,  %0, v16" ::"f"(f_[2]));
                asm volatile("vfmacc.vf v2,  %0, v16" ::"f"(f_[5]));
                asm volatile("vfmacc.vf v0,  %0, v16" ::"f"(f_[8]));


                asm volatile("vfmacc.vf v6,  %0, v18" ::"f"(f_[2]));
                asm volatile("vfmacc.vf v4,  %0, v18" ::"f"(f_[5]));
                asm volatile("vfmacc.vf v2,  %0, v18" ::"f"(f_[8]));


                asm volatile("vfmacc.vf v8,  %0, v20" ::"f"(f_[2]));
                asm volatile("vfmacc.vf v6,  %0, v20" ::"f"(f_[5]));
                asm volatile("vfmacc.vf v4,  %0, v20" ::"f"(f_[8]));


                asm volatile("vfmacc.vf v10,  %0, v22" ::"f"(f_[2]));
                asm volatile("vfmacc.vf v8,  %0, v22" ::"f"(f_[5]));
                asm volatile("vfmacc.vf v6,  %0, v22" ::"f"(f_[8]));


                asm volatile("vfmacc.vf v12,  %0, v24" ::"f"(f_[2]));
                asm volatile("vfmacc.vf v10,  %0, v24" ::"f"(f_[5]));
                asm volatile("vfmacc.vf v8,  %0, v24" ::"f"(f_[8]));


                asm volatile("vfmacc.vf v14,  %0, v26" ::"f"(f_[2]));
                asm volatile("vfmacc.vf v12,  %0, v26" ::"f"(f_[5]));
                asm volatile("vfmacc.vf v10,  %0, v26" ::"f"(f_[8]));


                }

            asm volatile("vsetvli zero, %0, e32, m2, ta, ma" ::"r"(vlen_out));


                i__ = i_ + r * C;
                f_ = f;
                
            asm volatile("vse32.v v0, (%0); add %0, %0, %1" : "+&r"(o_) : "r"(ldo));
            asm volatile("vse32.v v2, (%0); add %0, %0, %1" : "+&r"(o_) : "r"(ldo));
            asm volatile("vse32.v v4, (%0); add %0, %0, %1" : "+&r"(o_) : "r"(ldo));
            asm volatile("vse32.v v6, (%0); add %0, %0, %1" : "+&r"(o_) : "r"(ldo));
            asm volatile("vse32.v v8, (%0); add %0, %0, %1" : "+&r"(o_) : "r"(ldo));
            asm volatile("vse32.v v10, (%0); add %0, %0, %1" : "+&r"(o_) : "r"(ldo));

            asm volatile("vmv.v.v v0, v12");
            asm volatile("vmv.v.v v2, v14");

            }


        asm volatile("vsetvli zero, %0, e32, m2, ta, ma" ::"r"(vlen));

        if (last_group == 0)
        {
            asm volatile("vle32.v v16, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
            asm volatile("vle32.v v18, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
            asm volatile("vle32.v v20, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
            asm volatile("vle32.v v22, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
            asm volatile("vle32.v v24, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
            asm volatile("vle32.v v26, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(next_plane_(6)));

        }
        else if (last_group == 5)
        {
            asm volatile("vle32.v v16, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
            asm volatile("vle32.v v18, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
            asm volatile("vle32.v v20, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
            asm volatile("vle32.v v22, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
            asm volatile("vle32.v v24, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(next_plane_(5)));

        }
        else if (last_group == 4)
        {
            asm volatile("vle32.v v16, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
            asm volatile("vle32.v v18, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
            asm volatile("vle32.v v20, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
            asm volatile("vle32.v v22, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(next_plane_(4)));

        }
        else if (last_group == 3)
        {
            asm volatile("vle32.v v16, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
            asm volatile("vle32.v v18, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
            asm volatile("vle32.v v20, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(next_plane_(3)));

        }
        else if (last_group == 2)
        {
            asm volatile("vle32.v v16, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
            asm volatile("vle32.v v18, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(next_plane_(2)));

        }
        else
        {
            asm volatile("vle32.v v16, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(next_plane_(1)));

        }
        asm volatile("vfmacc.vf v0,  %0, v16" ::"f"(f_[6]));
        asm volatile("vfmacc.vf v2,  %0, v18" ::"f"(f_[6]));
        asm volatile("vfmul.vf v4,  v20, %0" :: "f"(f_[6]));
        asm volatile("vfmul.vf v6,  v22, %0" :: "f"(f_[6]));
        asm volatile("vfmul.vf v8,  v24, %0" :: "f"(f_[6]));
        asm volatile("vfmul.vf v10,  v26, %0" :: "f"(f_[6]));

        asm volatile("vslidedown.vi v26, v26, 1");
        asm volatile("vfmacc.vf v2,  %0, v16" ::"f"(f_[3]));
        asm volatile("vfmacc.vf v4,  %0, v18" ::"f"(f_[3]));
        asm volatile("vfmacc.vf v6,  %0, v20" ::"f"(f_[3]));
        asm volatile("vfmacc.vf v8,  %0, v22" ::"f"(f_[3]));
        asm volatile("vfmacc.vf v10,  %0, v24" ::"f"(f_[3]));

        asm volatile("vslidedown.vi v24, v24, 1");
        asm volatile("vfmacc.vf v4,  %0, v16" ::"f"(f_[0]));
        asm volatile("vfmacc.vf v6,  %0, v18" ::"f"(f_[0]));
        asm volatile("vfmacc.vf v8,  %0, v20" ::"f"(f_[0]));
        asm volatile("vfmacc.vf v10,  %0, v22" ::"f"(f_[0]));

        asm volatile("vslidedown.vi v22, v22, 1");
        asm volatile("vslidedown.vi v20, v20, 1");
        asm volatile("vslidedown.vi v18, v18, 1");
        asm volatile("vslidedown.vi v16, v16, 1");

        asm volatile("vfmacc.vf v0,  %0, v16" ::"f"(f_[7]));
        asm volatile("vfmacc.vf v2,  %0, v18" ::"f"(f_[7]));
        asm volatile("vfmacc.vf v4,  %0, v20" ::"f"(f_[7]));
        asm volatile("vfmacc.vf v6,  %0, v22" ::"f"(f_[7]));
        asm volatile("vfmacc.vf v8,  %0, v24" ::"f"(f_[7]));
        asm volatile("vfmacc.vf v10,  %0, v26" ::"f"(f_[7]));

        asm volatile("vslidedown.vi v26, v26, 1");
        asm volatile("vfmacc.vf v2,  %0, v16" ::"f"(f_[4]));
        asm volatile("vfmacc.vf v4,  %0, v18" ::"f"(f_[4]));
        asm volatile("vfmacc.vf v6,  %0, v20" ::"f"(f_[4]));
        asm volatile("vfmacc.vf v8,  %0, v22" ::"f"(f_[4]));
        asm volatile("vfmacc.vf v10,  %0, v24" ::"f"(f_[4]));

        asm volatile("vslidedown.vi v24, v24, 1");
        asm volatile("vfmacc.vf v4,  %0, v16" ::"f"(f_[1]));
        asm volatile("vfmacc.vf v6,  %0, v18" ::"f"(f_[1]));
        asm volatile("vfmacc.vf v8,  %0, v20" ::"f"(f_[1]));
        asm volatile("vfmacc.vf v10,  %0, v22" ::"f"(f_[1]));

        asm volatile("vslidedown.vi v22, v22, 1");
        asm volatile("vslidedown.vi v20, v20, 1");
        asm volatile("vslidedown.vi v18, v18, 1");
        asm volatile("vslidedown.vi v16, v16, 1");

        asm volatile("vfmacc.vf v0,  %0, v16" ::"f"(f_[8]));
        asm volatile("vfmacc.vf v2,  %0, v18" ::"f"(f_[8]));
        asm volatile("vfmacc.vf v4,  %0, v20" ::"f"(f_[8]));
        asm volatile("vfmacc.vf v6,  %0, v22" ::"f"(f_[8]));
        asm volatile("vfmacc.vf v8,  %0, v24" ::"f"(f_[8]));
        asm volatile("vfmacc.vf v10,  %0, v26" ::"f"(f_[8]));

        asm volatile("vfmacc.vf v2,  %0, v16" ::"f"(f_[5]));
        asm volatile("vfmacc.vf v4,  %0, v18" ::"f"(f_[5]));
        asm volatile("vfmacc.vf v6,  %0, v20" ::"f"(f_[5]));
        asm volatile("vfmacc.vf v8,  %0, v22" ::"f"(f_[5]));
        asm volatile("vfmacc.vf v10,  %0, v24" ::"f"(f_[5]));

        asm volatile("vfmacc.vf v4,  %0, v16" ::"f"(f_[2]));
        asm volatile("vfmacc.vf v6,  %0, v18" ::"f"(f_[2]));
        asm volatile("vfmacc.vf v8,  %0, v20" ::"f"(f_[2]));
        asm volatile("vfmacc.vf v10,  %0, v22" ::"f"(f_[2]));



        for(int ch = 1 ; ch < W ; ch ++){

            f_ += 9;

            if (last_group == 0)
            {
                asm volatile("vle32.v v16, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
                asm volatile("vle32.v v18, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
                asm volatile("vle32.v v20, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
                asm volatile("vle32.v v22, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
                asm volatile("vle32.v v24, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
                asm volatile("vle32.v v26, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(next_plane_(6)));

            }
            else if (last_group == 5)
            {
                asm volatile("vle32.v v16, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
                asm volatile("vle32.v v18, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
                asm volatile("vle32.v v20, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
                asm volatile("vle32.v v22, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
                asm volatile("vle32.v v24, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(next_plane_(5)));

            }
            else if (last_group == 4)
            {
                asm volatile("vle32.v v16, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
                asm volatile("vle32.v v18, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
                asm volatile("vle32.v v20, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
                asm volatile("vle32.v v22, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(next_plane_(4)));

            }
            else if (last_group == 3)
            {
                asm volatile("vle32.v v16, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
                asm volatile("vle32.v v18, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
                asm volatile("vle32.v v20, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(next_plane_(3)));

            }
            else if (last_group == 2)
            {
                asm volatile("vle32.v v16, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
                asm volatile("vle32.v v18, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(next_plane_(2)));

            }
            else
            {
                asm volatile("vle32.v v16, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(next_plane_(1)));

            }
            asm volatile("vfmacc.vf v0,  %0, v16" ::"f"(f_[6]));
            asm volatile("vfmacc.vf v2,  %0, v18" ::"f"(f_[6]));
            asm volatile("vfmacc.vf v4,  %0, v20" ::"f"(f_[6]));
            asm volatile("vfmacc.vf v6,  %0, v22" ::"f"(f_[6]));
            asm volatile("vfmacc.vf v8,  %0, v24" ::"f"(f_[6]));
            asm volatile("vfmacc.vf v10,  %0, v26" ::"f"(f_[6]));

            asm volatile("vslidedown.vi v26, v26, 1");
            asm volatile("vfmacc.vf v2,  %0, v16" ::"f"(f_[3]));
            asm volatile("vfmacc.vf v4,  %0, v18" ::"f"(f_[3]));
            asm volatile("vfmacc.vf v6,  %0, v20" ::"f"(f_[3]));
            asm volatile("vfmacc.vf v8,  %0, v22" ::"f"(f_[3]));
            asm volatile("vfmacc.vf v10,  %0, v24" ::"f"(f_[3]));

            asm volatile("vslidedown.vi v24, v24, 1");
            asm volatile("vfmacc.vf v4,  %0, v16" ::"f"(f_[0]));
            asm volatile("vfmacc.vf v6,  %0, v18" ::"f"(f_[0]));
            asm volatile("vfmacc.vf v8,  %0, v20" ::"f"(f_[0]));
            asm volatile("vfmacc.vf v10,  %0, v22" ::"f"(f_[0]));

            asm volatile("vslidedown.vi v22, v22, 1");
            asm volatile("vslidedown.vi v20, v20, 1");
            asm volatile("vslidedown.vi v18, v18, 1");
            asm volatile("vslidedown.vi v16, v16, 1");

            asm volatile("vfmacc.vf v0,  %0, v16" ::"f"(f_[7]));
            asm volatile("vfmacc.vf v2,  %0, v18" ::"f"(f_[7]));
            asm volatile("vfmacc.vf v4,  %0, v20" ::"f"(f_[7]));
            asm volatile("vfmacc.vf v6,  %0, v22" ::"f"(f_[7]));
            asm volatile("vfmacc.vf v8,  %0, v24" ::"f"(f_[7]));
            asm volatile("vfmacc.vf v10,  %0, v26" ::"f"(f_[7]));

            asm volatile("vslidedown.vi v26, v26, 1");
            asm volatile("vfmacc.vf v2,  %0, v16" ::"f"(f_[4]));
            asm volatile("vfmacc.vf v4,  %0, v18" ::"f"(f_[4]));
            asm volatile("vfmacc.vf v6,  %0, v20" ::"f"(f_[4]));
            asm volatile("vfmacc.vf v8,  %0, v22" ::"f"(f_[4]));
            asm volatile("vfmacc.vf v10,  %0, v24" ::"f"(f_[4]));

            asm volatile("vslidedown.vi v24, v24, 1");
            asm volatile("vfmacc.vf v4,  %0, v16" ::"f"(f_[1]));
            asm volatile("vfmacc.vf v6,  %0, v18" ::"f"(f_[1]));
            asm volatile("vfmacc.vf v8,  %0, v20" ::"f"(f_[1]));
            asm volatile("vfmacc.vf v10,  %0, v22" ::"f"(f_[1]));

            asm volatile("vslidedown.vi v22, v22, 1");
            asm volatile("vslidedown.vi v20, v20, 1");
            asm volatile("vslidedown.vi v18, v18, 1");
            asm volatile("vslidedown.vi v16, v16, 1");

            asm volatile("vfmacc.vf v0,  %0, v16" ::"f"(f_[8]));
            asm volatile("vfmacc.vf v2,  %0, v18" ::"f"(f_[8]));
            asm volatile("vfmacc.vf v4,  %0, v20" ::"f"(f_[8]));
            asm volatile("vfmacc.vf v6,  %0, v22" ::"f"(f_[8]));
            asm volatile("vfmacc.vf v8,  %0, v24" ::"f"(f_[8]));
            asm volatile("vfmacc.vf v10,  %0, v26" ::"f"(f_[8]));

            asm volatile("vfmacc.vf v2,  %0, v16" ::"f"(f_[5]));
            asm volatile("vfmacc.vf v4,  %0, v18" ::"f"(f_[5]));
            asm volatile("vfmacc.vf v6,  %0, v20" ::"f"(f_[5]));
            asm volatile("vfmacc.vf v8,  %0, v22" ::"f"(f_[5]));
            asm volatile("vfmacc.vf v10,  %0, v24" ::"f"(f_[5]));

            asm volatile("vfmacc.vf v4,  %0, v16" ::"f"(f_[2]));
            asm volatile("vfmacc.vf v6,  %0, v18" ::"f"(f_[2]));
            asm volatile("vfmacc.vf v8,  %0, v20" ::"f"(f_[2]));
            asm volatile("vfmacc.vf v10,  %0, v22" ::"f"(f_[2]));


            }

        asm volatile("vsetvli zero, %0, e32, m2, ta, ma" ::"r"(vlen_out));

        if (last_group == 0)
        {
        asm volatile("vse32.v v0, (%0); add %0, %0, %1" : "+&r"(o_) : "r"(ldo));
        asm volatile("vse32.v v2, (%0); add %0, %0, %1" : "+&r"(o_) : "r"(ldo));
        asm volatile("vse32.v v4, (%0); add %0, %0, %1" : "+&r"(o_) : "r"(ldo));
        asm volatile("vse32.v v6, (%0); add %0, %0, %1" : "+&r"(o_) : "r"(ldo));
        asm volatile("vse32.v v8, (%0); add %0, %0, %1" : "+&r"(o_) : "r"(ldo));
        asm volatile("vse32.v v10, (%0)" : "+&r"(o_));

        }
        else if (last_group == 5)
        {
        asm volatile("vse32.v v0, (%0); add %0, %0, %1" : "+&r"(o_) : "r"(ldo));
        asm volatile("vse32.v v2, (%0); add %0, %0, %1" : "+&r"(o_) : "r"(ldo));
        asm volatile("vse32.v v4, (%0); add %0, %0, %1" : "+&r"(o_) : "r"(ldo));
        asm volatile("vse32.v v6, (%0); add %0, %0, %1" : "+&r"(o_) : "r"(ldo));
        asm volatile("vse32.v v8, (%0)" : "+&r"(o_));

        }
        else if (last_group == 4)
        {
        asm volatile("vse32.v v0, (%0); add %0, %0, %1" : "+&r"(o_) : "r"(ldo));
        asm volatile("vse32.v v2, (%0); add %0, %0, %1" : "+&r"(o_) : "r"(ldo));
        asm volatile("vse32.v v4, (%0); add %0, %0, %1" : "+&r"(o_) : "r"(ldo));
        asm volatile("vse32.v v6, (%0)" : "+&r"(o_));

        }
        else if (last_group == 3)
        {
        asm volatile("vse32.v v0, (%0); add %0, %0, %1" : "+&r"(o_) : "r"(ldo));
        asm volatile("vse32.v v2, (%0); add %0, %0, %1" : "+&r"(o_) : "r"(ldo));
        asm volatile("vse32.v v4, (%0)" : "+&r"(o_));

        }
        else if (last_group == 2)
        {
        asm volatile("vse32.v v0, (%0); add %0, %0, %1" : "+&r"(o_) : "r"(ldo));
        asm volatile("vse32.v v2, (%0)" : "+&r"(o_));

        }
        else
        {
        asm volatile("vse32.v v0, (%0)" : "+&r"(o_));

        }
    }
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
