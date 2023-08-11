#include "custom_model_power_analysis.h"
#include "nn_functions.h"

void power_analysis() {

    #if defined(PA_MAT_VEC)
    // define vectors
    float weights[MAT_VEC_INPUT_SIZE*MAT_VEC_OUTPUT_SIZE];
    float bias[MAT_VEC_OUTPUT_SIZE];
    float mat_vec_input[MAT_VEC_INPUT_SIZE];
    float mat_vec_output[MAT_VEC_OUTPUT_SIZE];

    // run kernel
    float_mat_vec_product(MAT_VEC_OUTPUT_SIZE,
                          MAT_VEC_INPUT_SIZE,
                          weights,
                          bias,
                          mat_vec_input,
                          mat_vec_output);

    #elif defined(PA_CONV)
    // define vectors
    float conv_filter[CONV_FILTER_SIZE*CONV_FILTER_SIZE];
    float conv_input[CONV_INPUT_SIZE*CONV_INPUT_SIZE];
    float conv_output[(CONV_INPUT_SIZE-1)*(CONV_INPUT_SIZE-1)];
    float bias[1];

    // run kernel
    conv_2d(CONV_INPUT_SIZE,
            CONV_INPUT_SIZE,
            CONV_FILTER_SIZE,
            conv_filter,
            bias,
            conv_input,
            conv_output);

    #elif defined(PA_PAD)
    // define vectors
    float pad_input[PAD_INPUT_SIZE*PAD_INPUT_SIZE];
    float pad_output[(PAD_INPUT_SIZE-1)*(PAD_INPUT_SIZE-1)];

    // run kernel
    float_zero_pad(pad_output, pad_input, PAD_INPUT_SIZE, PAD_INPUT_SIZE, PAD_PADDING);

    #elif defined(PA_POOL)
    // define vectors
    float pool_input[POOL_INPUT_SIZE*POOL_INPUT_SIZE];
    float pool_output[(POOL_INPUT_SIZE/2)*(POOL_INPUT_SIZE/2)];

    // run kernel
    fmax_pool_vec_1xC_2x2(pool_output, pool_input, POOL_INPUT_SIZE, POOL_INPUT_SIZE, POOL_INPUT_CHANNEL, POOL_WINDOW_SIZE, POOL_STRIDE);


    #elif defined(PA_RELU)
    // define vectors
    float relu_input[RELU_INPUT_SIZE];
    float relu_output[RELU_INPUT_SIZE];

    // run kernel
    float_relu(RELU_INPUT_SIZE,
               relu_input,
               relu_output);

    #endif

}