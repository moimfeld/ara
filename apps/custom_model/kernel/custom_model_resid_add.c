#include "custom_model_resid_add.h"


void float_resid_add(float * output, float * input, float * residual, uint32_t len) {
    #if defined(USE_VEXT)
    int64_t remaining_len = len;
    int64_t vlen;
	while (remaining_len >0) {
        asm volatile("vsetvli %0, %1, e32, m8, ta, ma" :"=r"(vlen):"r"(remaining_len));
        asm volatile("vle32.v v8,  (%0); add %0, %0, %1" : "+&r"(input) : "r"(vlen));
	    asm volatile("vle32.v v16, (%0); add %0, %0, %1" : "+&r"(residual) : "r"(vlen));
        asm volatile("vfadd.vv v0, v8, v16");
        asm volatile("vse32.v v0, (%0); add %0, %0, %1" : "+&r"(output) : "r"(vlen));
        vlen = 1000;
        remaining_len -= vlen;
    }

    #else
    for (uint32_t i = 0; i < len; i++) {
        output[i] = input[i] + residual[i];
    }
    #endif
}