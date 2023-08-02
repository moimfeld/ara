/*
 * Copyright (C) 2010-2022 Arm Limited or its affiliates.
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the License); you may
 * not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an AS IS BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * Modifications copyright (C) 2021-2022 Chair of Electronic Design Automation, TUM
 */

#include "custom_model_forward_pass.h"
#include "custom_model_relu.h"
#include "custom_model_max_pool.h"
#include "custom_model_padding.h"
#include "custom_model_helper.h"
#include "custom_model_macros.h"
#include "nn_functions.h"
#include <stdint.h>
#include "util.h"
#include "runtime.h"


#if defined(TINY_FC_MODEL)
#include "tiny_fc_model.h"
#include "images_labels.h"
#elif defined(FC_MODEL)
#include "fc_model.h"
#include "images_labels.h"
#elif defined(CONV_MODEL)
#include "conv_model.h"
#include "images_labels.h"
#elif defined(CONV_POOL_MODEL)
#include "conv_pool_model.h"
#include "images_labels.h"
#elif defined(CONV_PAD_MODEL)
#include "conv_pad_model.h"
#include "images_labels.h"
#elif defined(RESNET)
#error Not Implemented :(
#else
#warning No model defined, defaulting to TINY_FC_MODEL. Use "-D <MODEL_NAME>" to define a model (Possible models TINY_FC_MODEL, FC_MODEL, CONV_MODEL, CONV_POOL_MODEL, CONV_PAD_MODEL, RESNET)
#include "tiny_fc_model.h"
#include "images_labels.h"
#endif


#if defined(USE_VEXT)
#include <riscv_vector.h>
#endif

#ifndef SPIKE
#include "printf.h"
#else
#include <stdio.h>
#endif


//   _____ ___ _   ___   __  _____ ____   __  __  ___  ____  _____ _     
//  |_   _|_ _| \ | \ \ / / |  ___/ ___| |  \/  |/ _ \|  _ \| ____| |    
//    | |  | ||  \| |\ V /  | |_ | |     | |\/| | | | | | | |  _| | |    
//    | |  | || |\  | | |   |  _|| |___  | |  | | |_| | |_| | |___| |___ 
//    |_| |___|_| \_| |_|   |_|   \____| |_|  |_|\___/|____/|_____|_____|


// Simulation of Ara
// =================


// Simulation running, end by pressing CTRL-c.
// Measuring enabled
// Using vectorized kernels
// Evaluating TINY_FC_MODEL
// MEASUREMENT: First (before ReLU) Fully-Connected Layer (in_dim=784, out_dim=64) execution took 431747 cycles
// MEASUREMENT: ReLU (dim=64) execution took 90 cycles (before ReLU)
// MEASUREMENT: First (after ReLU) Fully-Connected Layer (in_dim=784, out_dim=64) execution took 431837 cycles
// MEASUREMENT: Second (before ReLU) Fully-Connected Layer (in_dim=64, out_dim=32) execution took 19005 cycles
// MEASUREMENT: ReLU (dim=32) execution took 62 cycles (before ReLU)
// MEASUREMENT: Second (after ReLU) Fully-Connected Layer (in_dim=64, out_dim=32) execution took 19067 cycles
// MEASUREMENT: Third (before ReLU) Fully-Connected Layer (in_dim=32, out_dim=16) execution took 5206 cycles
// MEASUREMENT: ReLU (dim=16) execution took 62 cycles (before ReLU)
// MEASUREMENT: Third (after ReLU) Fully-Connected Layer (in_dim=32, out_dim=16) execution took 5268 cycles
// MEASUREMENT: Output Fully-Connected Layer (in_dim=16, out_dim=10) execution took 1930 cycles
// MEASUREMENT: Complete forward-pass took 458102 cycles
// MEASUREMENT: Argmax computation took 73 cyclesPrediction: 7
// Correct Label: 7
// [hw-cycles]:           0
// [1015614] -Info: ara_tb_verilator.sv:49: Assertion failed in TOP.ara_tb_verilator: Core Test *** SUCCESS *** (tohost = 0)
// - /home/moimfeld/workspace/socdaml/ara/hardware/tb/ara_tb_verilator.sv:52: Verilog $finish
// Received $finish() from Verilog, shutting down simulation.

// Simulation statistics
// =====================
// Executed cycles:  7bf9f
// Wallclock time:   120.898 s
// Simulation speed: 4200.29 cycles/s (4.20029 kHz)

#if defined(TINY_FC_MODEL)
uint8_t tiny_fc_model_forward(float * input)
{
    float dense_0_output[64];
    float relu_0_output[64];
    float dense_1_output[32];
    float relu_1_output[32];
    float dense_2_output[16];
    float relu_2_output[16];
    float dense_3_output[10];

    #if defined(MEASURE)
    start_timer();
    #endif

    // First Layer
    float_mat_vec_product(dense_0_weight_rows,
                            dense_0_weight_columns,
                            dense_0_weight,
                            dense_0_bias,
                            input,
                            dense_0_output);
    
    #if defined(MEASURE)
    stop_timer();
    int64_t pre_relu_0 = get_timer();
    printf("MEASUREMENT: First (before ReLU) Fully-Connected Layer (in_dim=%0ld, out_dim=%0ld) execution took %0ld cycles\n", dense_0_weight_columns, dense_0_weight_rows, pre_relu_0);
    start_timer();
    #endif

    float_relu(dense_0_weight_rows,
                dense_0_output,
                relu_0_output);

    #if defined(MEASURE)
    stop_timer();
    int64_t relu_0 = get_timer();
    printf("MEASUREMENT: ReLU (dim=%0ld) execution took %0ld cycles (before ReLU)\n", dense_0_weight_rows, relu_0);
    printf("MEASUREMENT: First (after ReLU) Fully-Connected Layer (in_dim=%0ld, out_dim=%0ld) execution took %0ld cycles\n", dense_0_weight_columns, dense_0_weight_rows, pre_relu_0 + relu_0);
    start_timer();
    #endif

    // Second Layer
    float_mat_vec_product(dense_1_weight_rows,
                            dense_1_weight_columns,
                            dense_1_weight,
                            dense_1_bias,
                            relu_0_output,
                            dense_1_output);
    #if defined(MEASURE)
    stop_timer();
    int64_t pre_relu_1 = get_timer();
    printf("MEASUREMENT: Second (before ReLU) Fully-Connected Layer (in_dim=%0ld, out_dim=%0ld) execution took %0ld cycles\n", dense_1_weight_columns, dense_1_weight_rows, pre_relu_1);
    start_timer();
    #endif

    float_relu(dense_1_weight_rows,
                dense_1_output,
                relu_1_output);

    #if defined(MEASURE)
    stop_timer();
    int64_t relu_1 = get_timer();
    printf("MEASUREMENT: ReLU (dim=%0ld) execution took %0ld cycles (before ReLU)\n", dense_1_weight_rows, relu_1);
    printf("MEASUREMENT: Second (after ReLU) Fully-Connected Layer (in_dim=%0ld, out_dim=%0ld) execution took %0ld cycles\n", dense_1_weight_columns, dense_1_weight_rows, pre_relu_1 + relu_1);
    start_timer();
    #endif


    // Third Layer
    float_mat_vec_product(dense_2_weight_rows,
                            dense_2_weight_columns,
                            dense_2_weight,
                            dense_2_bias,
                            relu_1_output,
                            dense_2_output);

    #if defined(MEASURE)
    stop_timer();
    int64_t pre_relu_2 = get_timer();
    printf("MEASUREMENT: Third (before ReLU) Fully-Connected Layer (in_dim=%0ld, out_dim=%0ld) execution took %0ld cycles\n", dense_2_weight_columns, dense_2_weight_rows, pre_relu_2);
    start_timer();
    #endif

    float_relu(dense_2_weight_rows,
                dense_2_output,
                relu_2_output);

    #if defined(MEASURE)
    stop_timer();
    int64_t relu_2 = get_timer();
    printf("MEASUREMENT: ReLU (dim=%0ld) execution took %0ld cycles (before ReLU)\n", dense_2_weight_rows, relu_2);
    printf("MEASUREMENT: Third (after ReLU) Fully-Connected Layer (in_dim=%0ld, out_dim=%0ld) execution took %0ld cycles\n", dense_2_weight_columns, dense_2_weight_rows, pre_relu_2 + relu_2);
    start_timer();
    #endif


    // Output Layer
    float_mat_vec_product(dense_3_weight_rows,
                            dense_3_weight_columns,
                            dense_3_weight,
                            dense_3_bias,
                            relu_2_output,
                            dense_3_output);
    
    #if defined(MEASURE)
    stop_timer();
    int64_t pre_relu_3 = get_timer();
    printf("MEASUREMENT: Output Fully-Connected Layer (in_dim=%0ld, out_dim=%0ld) execution took %0ld cycles\n", dense_3_weight_columns, dense_3_weight_rows, pre_relu_3);
    printf("MEASUREMENT: Complete forward-pass took %0ld cycles\n", pre_relu_0 + relu_0 + pre_relu_1 + relu_1 + pre_relu_2 + relu_2 + pre_relu_3);
    start_timer();
    #endif

    uint32_t argmax = float_argmax(10, dense_3_output);

    #if defined(MEASURE)
    stop_timer();
    int64_t argmax_time = get_timer();
    printf("MEASUREMENT: Argmax computation took %0ld cycles\n", argmax_time);
    #endif

    return argmax;
}
#endif


//   _____ ____   __  __  ___  ____  _____ _     
//  |  ___/ ___| |  \/  |/ _ \|  _ \| ____| |    
//  | |_ | |     | |\/| | | | | | | |  _| | |    
//  |  _|| |___  | |  | | |_| | |_| | |___| |___ 
//  |_|   \____| |_|  |_|\___/|____/|_____|_____|

// Simulation of Ara
// =================


// Simulation running, end by pressing CTRL-c.
// Measuring enabled
// Using vectorized kernels
// Evaluating FC_MODEL
// MEASUREMENT: First (before ReLU) Fully-Connected Layer (in_dim=784, out_dim=256) execution took 1725993 cycles
// MEASUREMENT: ReLU (dim=256) execution took 268 cycles (before ReLU)
// MEASUREMENT: First (after ReLU) Fully-Connected Layer (in_dim=784, out_dim=256) execution took 1726261 cycles
// MEASUREMENT: Second (before ReLU) Fully-Connected Layer (in_dim=256, out_dim=128) execution took 285730 cycles
// MEASUREMENT: ReLU (dim=128) execution took 151 cycles (before ReLU)
// MEASUREMENT: Second (after ReLU) Fully-Connected Layer (in_dim=256, out_dim=128) execution took 285881 cycles
// MEASUREMENT: Third (before ReLU) Fully-Connected Layer (in_dim=128, out_dim=64) execution took 72958 cycles
// MEASUREMENT: ReLU (dim=64) execution took 88 cycles (before ReLU)
// MEASUREMENT: Third (after ReLU) Fully-Connected Layer (in_dim=128, out_dim=64) execution took 73046 cycles
// MEASUREMENT: Output Fully-Connected Layer (in_dim=64, out_dim=10) execution took 6023 cycles
// MEASUREMENT: Complete forward-pass took 2091211 cycles
// MEASUREMENT: Argmax computation took 70 cycles
// Prediction: 7
// Correct Label: 7
// [hw-cycles]:           0
// [4281592] -Info: ara_tb_verilator.sv:49: Assertion failed in TOP.ara_tb_verilator: Core Test *** SUCCESS *** (tohost = 0)
// - /home/moimfeld/workspace/socdaml/ara/hardware/tb/ara_tb_verilator.sv:52: Verilog $finish
// Received $finish() from Verilog, shutting down simulation.

// Simulation statistics
// =====================
// Executed cycles:  20aa7c
// Wallclock time:   492.158 s
// Simulation speed: 4349.81 cycles/s (4.34981 kHz)

#if defined(FC_MODEL)
uint8_t fc_model_forward(float * input)
{
    float dense_0_output[256];
    float relu_0_output[256];
    float dense_1_output[128];
    float relu_1_output[128];
    float dense_2_output[64];
    float relu_2_output[64];
    float dense_3_output[10];

    // First Layer
    #if defined(MEASURE)
    start_timer();
    #endif

    float_mat_vec_product(dense_0_weight_rows,
                            dense_0_weight_columns,
                            dense_0_weight,
                            dense_0_bias,
                            input,
                            dense_0_output);

    #if defined(MEASURE)
    stop_timer();
    int64_t pre_relu_0 = get_timer();
    printf("MEASUREMENT: First (before ReLU) Fully-Connected Layer (in_dim=%0ld, out_dim=%0ld) execution took %0ld cycles\n", dense_0_weight_columns, dense_0_weight_rows, pre_relu_0);
    start_timer();
    #endif

    float_relu(dense_0_weight_rows,
                dense_0_output,
                relu_0_output);

    #if defined(MEASURE)
    stop_timer();
    int64_t relu_0 = get_timer();
    printf("MEASUREMENT: ReLU (dim=%0ld) execution took %0ld cycles (before ReLU)\n", dense_0_weight_rows, relu_0);
    printf("MEASUREMENT: First (after ReLU) Fully-Connected Layer (in_dim=%0ld, out_dim=%0ld) execution took %0ld cycles\n", dense_0_weight_columns, dense_0_weight_rows, pre_relu_0 + relu_0);
    start_timer();
    #endif

    // Second Layer
    float_mat_vec_product(dense_1_weight_rows,
                            dense_1_weight_columns,
                            dense_1_weight,
                            dense_1_bias,
                            relu_0_output,
                            dense_1_output);
    
    #if defined(MEASURE)
    stop_timer();
    int64_t pre_relu_1 = get_timer();
    printf("MEASUREMENT: Second (before ReLU) Fully-Connected Layer (in_dim=%0ld, out_dim=%0ld) execution took %0ld cycles\n", dense_1_weight_columns, dense_1_weight_rows, pre_relu_1);
    start_timer();
    #endif

    float_relu(dense_1_weight_rows,
                dense_1_output,
                relu_1_output);
    
    #if defined(MEASURE)
    stop_timer();
    int64_t relu_1 = get_timer();
    printf("MEASUREMENT: ReLU (dim=%0ld) execution took %0ld cycles (before ReLU)\n", dense_1_weight_rows, relu_1);
    printf("MEASUREMENT: Second (after ReLU) Fully-Connected Layer (in_dim=%0ld, out_dim=%0ld) execution took %0ld cycles\n", dense_1_weight_columns, dense_1_weight_rows, pre_relu_1 + relu_1);
    start_timer();
    #endif

    // Third Layer
    float_mat_vec_product(dense_2_weight_rows,
                            dense_2_weight_columns,
                            dense_2_weight,
                            dense_2_bias,
                            relu_1_output,
                            dense_2_output);

    #if defined(MEASURE)
    stop_timer();
    int64_t pre_relu_2 = get_timer();
    printf("MEASUREMENT: Third (before ReLU) Fully-Connected Layer (in_dim=%0ld, out_dim=%0ld) execution took %0ld cycles\n", dense_2_weight_columns, dense_2_weight_rows, pre_relu_2);
    start_timer();
    #endif

    float_relu(dense_2_weight_rows,
                dense_2_output,
                relu_2_output);
    
    #if defined(MEASURE)
    stop_timer();
    int64_t relu_2 = get_timer();
    printf("MEASUREMENT: ReLU (dim=%0ld) execution took %0ld cycles (before ReLU)\n", dense_2_weight_rows, relu_2);
    printf("MEASUREMENT: Third (after ReLU) Fully-Connected Layer (in_dim=%0ld, out_dim=%0ld) execution took %0ld cycles\n", dense_2_weight_columns, dense_2_weight_rows, pre_relu_2 + relu_2);
    start_timer();
    #endif

    // Output Layer
    float_mat_vec_product(dense_3_weight_rows,
                            dense_3_weight_columns,
                            dense_3_weight,
                            dense_3_bias,
                            relu_2_output,
                            dense_3_output);

    #if defined(MEASURE)
    stop_timer();
    int64_t pre_relu_3 = get_timer();
    printf("MEASUREMENT: Output Fully-Connected Layer (in_dim=%0ld, out_dim=%0ld) execution took %0ld cycles\n", dense_3_weight_columns, dense_3_weight_rows, pre_relu_3);
    printf("MEASUREMENT: Complete forward-pass took %0ld cycles\n", pre_relu_0 + relu_0 + pre_relu_1 + relu_1 + pre_relu_2 + relu_2 + pre_relu_3);
    start_timer();
    #endif


    uint32_t argmax = float_argmax(10, dense_3_output);

    #if defined(MEASURE)
    stop_timer();
    int64_t argmax_time = get_timer();
    printf("MEASUREMENT: Argmax computation took %0ld cycles\n", argmax_time);
    #endif

    return argmax;
}
#endif


//    ____ ___  _   ___     __  __  __  ___  ____  _____ _     
//   / ___/ _ \| \ | \ \   / / |  \/  |/ _ \|  _ \| ____| |    
//  | |  | | | |  \| |\ \ / /  | |\/| | | | | | | |  _| | |    
//  | |__| |_| | |\  | \ V /   | |  | | |_| | |_| | |___| |___ 
//   \____\___/|_| \_|  \_/    |_|  |_|\___/|____/|_____|_____|

// Simulation of Ara
// =================


// Simulation running, end by pressing CTRL-c.
// Measuring enabled
// Using vectorized kernels
// Evaluating CONV_MODEL
// MEASUREMENT: First (before ReLU) Convolutional Layer (in_dim=28x28, filter=3) execution took 3260 cycles
// MEASUREMENT: ReLU (dim=26x26) execution took 646 cycles (before ReLU)
// MEASUREMENT: First (after ReLU) Convolutional Layer (in_dim=28x28, filter=3) execution took 3906 cycles
// MEASUREMENT: Second (before ReLU) Convolutional Layer (in_dim=26x26, filter=3) execution took 2531 cycles
// MEASUREMENT: ReLU (dim=24x24) execution took 547 cycles (before ReLU)
// MEASUREMENT: Second (after ReLU) Convolutional Layer (in_dim=26x26, filter=3) execution took 3078 cycles
// MEASUREMENT: Third (before ReLU) Convolutional Layer (in_dim=24x24, filter=3) execution took 2276 cycles
// MEASUREMENT: ReLU (dim=22x22) execution took 465 cycles (before ReLU)
// MEASUREMENT: Third (after ReLU) Convolutional Layer (in_dim=24x24, filter=3) execution took 2741 cycles
// MEASUREMENT: Output Fully-Connected Layer (in_dim=484, out_dim=10) execution took 42056 cycles
// MEASUREMENT: Complete forward-pass took 51781 cycles
// MEASUREMENT: Argmax computation took 70 cycles
// Prediction: 7
// Correct Label: 7
// [hw-cycles]:           0
// [209518] -Info: ara_tb_verilator.sv:49: Assertion failed in TOP.ara_tb_verilator: Core Test *** SUCCESS *** (tohost = 0)
// - /home/moimfeld/workspace/socdaml/ara/hardware/tb/ara_tb_verilator.sv:52: Verilog $finish
// Received $finish() from Verilog, shutting down simulation.

// Simulation statistics
// =====================
// Executed cycles:  19937
// Wallclock time:   30.353 s
// Simulation speed: 3451.36 cycles/s (3.45136 kHz)

#if defined(CONV_MODEL)
uint8_t conv_model_forward(float * input)
{
    float output_conv0[26*26];
    float output_relu0[26*26];
    float output_conv1[24*24];
    float output_relu1[24*24];
    float output_conv2[22*22];
    float output_relu2[22*22];
    float output[10];

    // First layer
    #if defined(MEASURE)
    start_timer();
    #endif

    conv_2d(28,
            28,
            conv_0_weight_rows,
            conv_0_weight,
            conv_0_bias,
            input,
            output_conv0);

    #if defined(MEASURE)
    stop_timer();
    int64_t pre_relu_0 = get_timer();
    printf("MEASUREMENT: First (before ReLU) Convolutional Layer (in_dim=%0ldx%0ld, filter=%0ld) execution took %0ld cycles\n", 28, 28, conv_0_weight_rows, pre_relu_0);
    start_timer();
    #endif

    DEBUG_PRINTF_NOARGS("conv_model_forward: First conv done\n");
    float_relu(26*26,
               output_conv0,
               output_relu0);
    
    #if defined(MEASURE)
    stop_timer();
    int64_t relu_0 = get_timer();
    printf("MEASUREMENT: ReLU (dim=%0ldx%0ld) execution took %0ld cycles (before ReLU)\n", 26, 26, relu_0);
    printf("MEASUREMENT: First (after ReLU) Convolutional Layer (in_dim=%0ldx%0ld, filter=%0ld) execution took %0ld cycles\n", 28, 28, conv_0_weight_rows, pre_relu_0 + relu_0);
    start_timer();
    #endif
    
    // second layer
    conv_2d(26,
            26,
            conv_1_weight_rows,
            conv_1_weight,
            conv_1_bias,
            output_relu0,
            output_conv1);
    
    #if defined(MEASURE)
    stop_timer();
    int64_t pre_relu_1 = get_timer();
    printf("MEASUREMENT: Second (before ReLU) Convolutional Layer (in_dim=%0ldx%0ld, filter=%0ld) execution took %0ld cycles\n", 26, 26, conv_1_weight_rows, pre_relu_1);
    start_timer();
    #endif

    DEBUG_PRINTF_NOARGS("conv_model_forward: Second conv done\n");
    float_relu(24*24,
               output_conv1,
               output_relu1);

    #if defined(MEASURE)
    stop_timer();
    int64_t relu_1 = get_timer();
    printf("MEASUREMENT: ReLU (dim=%0ldx%0ld) execution took %0ld cycles (before ReLU)\n", 24, 24, relu_1);
    printf("MEASUREMENT: Second (after ReLU) Convolutional Layer (in_dim=%0ldx%0ld, filter=%0ld) execution took %0ld cycles\n", 26, 26, conv_1_weight_rows, pre_relu_1 + relu_1);
    start_timer();
    #endif

    // Third layer
    conv_2d(24,
            24,
            conv_2_weight_rows,
            conv_2_weight,
            conv_2_bias,
            output_relu1,
            output_conv2);

    #if defined(MEASURE)
    stop_timer();
    int64_t pre_relu_2 = get_timer();
    printf("MEASUREMENT: Third (before ReLU) Convolutional Layer (in_dim=%0ldx%0ld, filter=%0ld) execution took %0ld cycles\n", 24, 24, conv_2_weight_rows, pre_relu_2);
    start_timer();
    #endif

    DEBUG_PRINTF_NOARGS("conv_model_forward: Third conv done\n");

    #if defined(DEBUG)
    for (uint32_t i = 0; i < 22; i++) {
        for (uint32_t j = 0; j < 22; j++) {
            char resultStr[8];
            getFloatUpTo5thDecimal(output_conv2[i * 26 + j], resultStr);
            printf("%s ", resultStr);
        }
        printf("\n");
    }
    #endif

    float_relu(22*22,
               output_conv2,
               output_relu2);

    #if defined(MEASURE)
    stop_timer();
    int64_t relu_2 = get_timer();
    printf("MEASUREMENT: ReLU (dim=%0ldx%0ld) execution took %0ld cycles (before ReLU)\n", 22, 22, relu_2);
    printf("MEASUREMENT: Third (after ReLU) Convolutional Layer (in_dim=%0ldx%0ld, filter=%0ld) execution took %0ld cycles\n", 24, 24, conv_2_weight_rows, pre_relu_2 + relu_2);
    start_timer();
    #endif

    float_mat_vec_product(dense_0_weight_rows,
                          dense_0_weight_columns,
                          dense_0_weight,
                          dense_0_bias,
                          output_relu2,
                          output);

    #if defined(MEASURE)
    stop_timer();
    int64_t pre_relu_3 = get_timer();
    printf("MEASUREMENT: Output Fully-Connected Layer (in_dim=%0ld, out_dim=%0ld) execution took %0ld cycles\n", dense_0_weight_columns, dense_0_weight_rows, pre_relu_3);
    printf("MEASUREMENT: Complete forward-pass took %0ld cycles\n", pre_relu_0 + relu_0 + pre_relu_1 + relu_1 + pre_relu_2 + relu_2 + pre_relu_3);
    start_timer();
    #endif

    #if defined(DEBUG)
    for (uint32_t i = 0; i < 10; i++) {
        char resultStr[8];
        getFloatUpTo5thDecimal(output[i], resultStr);
        printf("Formatted result: %s\n", resultStr);
    }
    #endif

    uint32_t argmax = float_argmax(10, output);

    #if defined(MEASURE)
    stop_timer();
    int64_t argmax_time = get_timer();
    printf("MEASUREMENT: Argmax computation took %0ld cycles\n", argmax_time);
    #endif

    return argmax;
}
#endif


//    ____ ___  _   ___     __  ____   ___   ___  _       __  __  ___  ____  _____ _     
//   / ___/ _ \| \ | \ \   / / |  _ \ / _ \ / _ \| |     |  \/  |/ _ \|  _ \| ____| |    
//  | |  | | | |  \| |\ \ / /  | |_) | | | | | | | |     | |\/| | | | | | | |  _| | |    
//  | |__| |_| | |\  | \ V /   |  __/| |_| | |_| | |___  | |  | | |_| | |_| | |___| |___ 
//   \____\___/|_| \_|  \_/    |_|    \___/ \___/|_____| |_|  |_|\___/|____/|_____|_____|


// Simulation of Ara
// =================


// Simulation running, end by pressing CTRL-c.
// Measuring enabled
// Using vectorized kernels
// Evaluating CONV_POOL_MODEL
// MEASUREMENT: First (before ReLU) Convolutional Layer (in_dim=28x28, filter=3) execution took 3234 cycles
// MEASUREMENT: ReLU (dim=26x26) execution took 646 cycles (before ReLU)
// MEASUREMENT: First (after ReLU) Convolutional Layer (in_dim=28x28, filter=3) execution took 3880 cycles
// MEASUREMENT: Second (before ReLU) Convolutional Layer (in_dim=26x26, filter=3) execution took 2531 cycles
// MEASUREMENT: ReLU (dim=24x24) execution took 539 cycles (before ReLU)
// MEASUREMENT: Second (after ReLU) Convolutional Layer (in_dim=26x26, filter=3) execution took 3070 cycles
// MEASUREMENT: Pooling Layer (in_dim=24x24, window=2x2, stride=2) execution took 2531 cycles
// MEASUREMENT: Output Fully-Connected Layer (in_dim=144, out_dim=10) execution took 12942 cycles
// MEASUREMENT: Complete forward-pass took 21326 cycles
// MEASUREMENT: Argmax computation took 69 cycles
// Prediction: 7
// Correct Label: 7
// [hw-cycles]:           0
// [132382] -Info: ara_tb_verilator.sv:49: Assertion failed in TOP.ara_tb_verilator: Core Test *** SUCCESS *** (tohost = 0)
// - /home/moimfeld/workspace/socdaml/ara/hardware/tb/ara_tb_verilator.sv:52: Verilog $finish
// Received $finish() from Verilog, shutting down simulation.

// Simulation statistics
// =====================
// Executed cycles:  1028f
// Wallclock time:   20.509 s
// Simulation speed: 3227.41 cycles/s (3.22741 kHz)

#if defined(CONV_POOL_MODEL)
uint8_t conv_pool_model_forward(float * input)
{
    float output_conv0[26*26];
    float output_relu0[26*26];
    float output_conv1[24*24];
    float output_relu1[24*24];
    float output_maxPool[12*12];
    float output[10];

    // First Layer
    #if defined(MEASURE)
    start_timer();
    #endif

    conv_2d(28,
            28,
            conv_0_weight_rows,
            conv_0_weight,
            conv_0_bias,
            input,
            output_conv0);

    #if defined(MEASURE)
    stop_timer();
    int64_t pre_relu_0 = get_timer();
    printf("MEASUREMENT: First (before ReLU) Convolutional Layer (in_dim=%0ldx%0ld, filter=%0ld) execution took %0ld cycles\n", 28, 28, conv_0_weight_rows, pre_relu_0);
    start_timer();
    #endif

    DEBUG_PRINTF_NOARGS("conv_model_forward: First conv done\n");
    float_relu(26*26,
               output_conv0,
               output_relu0);

    #if defined(MEASURE)
    stop_timer();
    int64_t relu_0 = get_timer();
    printf("MEASUREMENT: ReLU (dim=%0ldx%0ld) execution took %0ld cycles (before ReLU)\n", 26, 26, relu_0);
    printf("MEASUREMENT: First (after ReLU) Convolutional Layer (in_dim=%0ldx%0ld, filter=%0ld) execution took %0ld cycles\n", 28, 28, conv_0_weight_rows, pre_relu_0 + relu_0);
    start_timer();
    #endif

    conv_2d(26,
            26,
            conv_1_weight_rows,
            conv_1_weight,
            conv_1_bias,
            output_relu0,
            output_conv1);

    #if defined(MEASURE)
    stop_timer();
    int64_t pre_relu_1 = get_timer();
    printf("MEASUREMENT: Second (before ReLU) Convolutional Layer (in_dim=%0ldx%0ld, filter=%0ld) execution took %0ld cycles\n", 26, 26, conv_1_weight_rows, pre_relu_1);
    start_timer();
    #endif

    DEBUG_PRINTF_NOARGS("conv_model_forward: Second conv done\n");
    float_relu(24*24,
               output_conv1,
               output_relu1);

    #if defined(MEASURE)
    stop_timer();
    int64_t relu_1 = get_timer();
    printf("MEASUREMENT: ReLU (dim=%0ldx%0ld) execution took %0ld cycles (before ReLU)\n", 24, 24, relu_1);
    printf("MEASUREMENT: Second (after ReLU) Convolutional Layer (in_dim=%0ldx%0ld, filter=%0ld) execution took %0ld cycles\n", 26, 26, conv_1_weight_rows, pre_relu_1 + relu_1);
    start_timer();
    #endif

    #if defined(DEBUG)
    for (uint32_t i = 0; i < 24; i++) {
        for (uint32_t j = 0; j < 24; j++) {
            char resultStr[8];
            getFloatUpTo5thDecimal(output_relu1[i * 24 + j], resultStr);
            printf("%s ", resultStr);
        }
        printf("\n");
    }
    #endif


    // output, input, rows, columns, channels, pool window size, stride
    fmax_pool_vec_1xC_2x2(output_maxPool, output_relu1, 24, 24, 1, 2, 2);

    #if defined(MEASURE)
    stop_timer();
    int64_t pooling_time = get_timer();
    printf("MEASUREMENT: Pooling Layer (in_dim=%0ldx%0ld, window=%0ldx%0ld, stride=%0ld) execution took %0ld cycles\n", 24, 24, 2, 2, 2, pre_relu_1);
    start_timer();
    #endif


    float_mat_vec_product(dense_0_weight_rows,
                          dense_0_weight_columns,
                          dense_0_weight,
                          dense_0_bias,
                          output_maxPool,
                          output);

    #if defined(MEASURE)
    stop_timer();
    int64_t pre_relu_2 = get_timer();
    printf("MEASUREMENT: Output Fully-Connected Layer (in_dim=%0ld, out_dim=%0ld) execution took %0ld cycles\n", dense_0_weight_columns, dense_0_weight_rows, pre_relu_2);
    printf("MEASUREMENT: Complete forward-pass took %0ld cycles\n", pre_relu_0 + relu_0 + pre_relu_1 + relu_1 + pooling_time + pre_relu_2);
    start_timer();
    #endif

    #if defined(DEBUG)
    for (uint32_t i = 0; i < 10; i++) {
        char resultStr[8];
        getFloatUpTo5thDecimal(output[i], resultStr);
        printf("Formatted result: %s\n", resultStr);
    }
    #endif

    uint32_t argmax = float_argmax(10, output);

    #if defined(MEASURE)
    stop_timer();
    int64_t argmax_time = get_timer();
    printf("MEASUREMENT: Argmax computation took %0ld cycles\n", argmax_time);
    #endif

    return argmax;
}
#endif


//    ____ ___  _   ___     __  ____   _    ____    __  __  ___  ____  _____ _     
//   / ___/ _ \| \ | \ \   / / |  _ \ / \  |  _ \  |  \/  |/ _ \|  _ \| ____| |    
//  | |  | | | |  \| |\ \ / /  | |_) / _ \ | | | | | |\/| | | | | | | |  _| | |    
//  | |__| |_| | |\  | \ V /   |  __/ ___ \| |_| | | |  | | |_| | |_| | |___| |___ 
//   \____\___/|_| \_|  \_/    |_| /_/   \_\____/  |_|  |_|\___/|____/|_____|_____|


// Simulation of Ara
// =================


// Simulation running, end by pressing CTRL-c.
// Measuring enabled
// Using vectorized kernels
// Evaluating CONV_PAD_MODEL
// MEASUREMENT: Padding Layer (in_dim=28x28, padding=1) execution took 2336 cycles
// MEASUREMENT: First (before ReLU) Convolutional Layer (in_dim=30x30, filter=3) execution took 3210 cycles
// MEASUREMENT: ReLU (dim=28x28) execution took 440 cycles (before ReLU)
// MEASUREMENT: First (after ReLU) Convolutional Layer (in_dim=30x30, filter=3) execution took 3650 cycles
// MEASUREMENT: Padding Layer (in_dim=28x28, padding=1) execution took 2157 cycles
// MEASUREMENT: Second (before ReLU) Convolutional Layer (in_dim=30x30, filter=3) execution took 2977 cycles
// MEASUREMENT: ReLU (dim=28x28) execution took 447 cycles (before ReLU)
// MEASUREMENT: Second (after ReLU) Convolutional Layer (in_dim=30x30, filter=3) execution took 3424 cycles
// MEASUREMENT: Padding Layer (in_dim=28x28, padding=1) execution took 2155 cycles
// MEASUREMENT: Third (before ReLU) Convolutional Layer (in_dim=30x30, filter=3) execution took 2942 cycles
// MEASUREMENT: ReLU (dim=28x28) execution took 431 cycles (before ReLU)
// MEASUREMENT: Third (after ReLU) Convolutional Layer (in_dim=30x30, filter=3) execution took 3373 cycles
// MEASUREMENT: Output Fully-Connected Layer (in_dim=784, out_dim=10) execution took 67653 cycles
// MEASUREMENT: Complete forward-pass took 84748 cycles
// MEASUREMENT: Argmax computation took 71 cycles
// Prediction: 7
// Correct Label: 7
// [hw-cycles]:           0
// [300582] -Info: ara_tb_verilator.sv:49: Assertion failed in TOP.ara_tb_verilator: Core Test *** SUCCESS *** (tohost = 0)
// - /home/moimfeld/workspace/socdaml/ara/hardware/tb/ara_tb_verilator.sv:52: Verilog $finish
// Received $finish() from Verilog, shutting down simulation.

// Simulation statistics
// =====================
// Executed cycles:  24b13
// Wallclock time:   42.884 s
// Simulation speed: 3504.59 cycles/s (3.50459 kHz)

#if defined(CONV_PAD_MODEL)
uint8_t conv_pad_model_forward(float * input)
{
    float output_pad0[30*30];
    float output_conv0[28*28];
    float output_relu0[28*28];
    float output_pad1[30*30];
    float output_conv1[28*28];
    float output_relu1[28*28];
    float output_pad2[30*30];
    float output_conv2[28*28];
    float output_relu2[28*28];
    float output[10];


    // first layer
    #if defined(MEASURE)
    start_timer();
    #endif

    // output, input, i_rows, i_columns, padding
    float_zero_pad(output_pad0, input, 28, 28, 1);

    #if defined(MEASURE)
    stop_timer();
    int64_t padding_0 = get_timer();
    printf("MEASUREMENT: Padding Layer (in_dim=%0ldx%0ld, padding=%0ld) execution took %0ld cycles\n", 28, 28, 1, padding_0);
    start_timer();
    #endif

    #if defined(DEBUG)
    for (uint32_t i = 0; i < 30; i++) {
      for (uint32_t j = 0; j < 30; j++) {
          char resultStr[8];
          getFloatUpTo5thDecimal(output_pad0[i * 30 + j], resultStr);
          printf("%s ", resultStr);
      }
      printf("\n");
    }
    #endif

    conv_2d(30,
            30,
            conv_0_weight_rows,
            conv_0_weight,
            conv_0_bias,
            output_pad0,
            output_conv0);
    
    #if defined(MEASURE)
    stop_timer();
    int64_t pre_relu_0 = get_timer();
    printf("MEASUREMENT: First (before ReLU) Convolutional Layer (in_dim=%0ldx%0ld, filter=%0ld) execution took %0ld cycles\n", 30, 30, conv_0_weight_rows, pre_relu_0);
    start_timer();
    #endif

    DEBUG_PRINTF_NOARGS("conv_model_forward: First conv done\n");
    float_relu(28*28,
               output_conv0,
               output_relu0);

    #if defined(MEASURE)
    stop_timer();
    int64_t relu_0 = get_timer();
    printf("MEASUREMENT: ReLU (dim=%0ldx%0ld) execution took %0ld cycles (before ReLU)\n", 28, 28, relu_0);
    printf("MEASUREMENT: First (after ReLU) Convolutional Layer (in_dim=%0ldx%0ld, filter=%0ld) execution took %0ld cycles\n", 30, 30, conv_0_weight_rows, pre_relu_0 + relu_0);
    start_timer();
    #endif


    // second layer
    float_zero_pad(output_pad1, output_relu0, 28, 28, 1);

    #if defined(MEASURE)
    stop_timer();
    int64_t padding_1 = get_timer();
    printf("MEASUREMENT: Padding Layer (in_dim=%0ldx%0ld, padding=%0ld) execution took %0ld cycles\n", 28, 28, 1, padding_1);
    start_timer();
    #endif

    #if defined(DEBUG)
    for (uint32_t i = 0; i < 30; i++) {
        for (uint32_t j = 0; j < 30; j++) {
            char resultStr[8];
            getFloatUpTo5thDecimal(output_pad1[i * 30 + j], resultStr);
            printf("%s ", resultStr);
        }
        printf("\n");
    }
    #endif
    conv_2d(30,
            30,
            conv_1_weight_rows,
            conv_1_weight,
            conv_1_bias,
            output_relu0,
            output_conv1);
        
    #if defined(MEASURE)
    stop_timer();
    int64_t pre_relu_1 = get_timer();
    printf("MEASUREMENT: Second (before ReLU) Convolutional Layer (in_dim=%0ldx%0ld, filter=%0ld) execution took %0ld cycles\n", 30, 30, conv_1_weight_rows, pre_relu_1);
    start_timer();
    #endif

    DEBUG_PRINTF_NOARGS("conv_model_forward: Second conv done\n");
    float_relu(28*28,
               output_conv1,
               output_relu1);

    #if defined(MEASURE)
    stop_timer();
    int64_t relu_1 = get_timer();
    printf("MEASUREMENT: ReLU (dim=%0ldx%0ld) execution took %0ld cycles (before ReLU)\n", 28, 28, relu_1);
    printf("MEASUREMENT: Second (after ReLU) Convolutional Layer (in_dim=%0ldx%0ld, filter=%0ld) execution took %0ld cycles\n", 30, 30, conv_1_weight_rows, pre_relu_1 + relu_1);
    start_timer();
    #endif

    // third layer
    float_zero_pad(output_pad2, output_relu1, 28, 28, 1);

    #if defined(MEASURE)
    stop_timer();
    int64_t padding_2 = get_timer();
    printf("MEASUREMENT: Padding Layer (in_dim=%0ldx%0ld, padding=%0ld) execution took %0ld cycles\n", 28, 28, 1, padding_2);
    start_timer();
    #endif

    conv_2d(30,
            30,
            conv_2_weight_rows,
            conv_2_weight,
            conv_2_bias,
            output_relu1,
            output_conv2);

    #if defined(MEASURE)
    stop_timer();
    int64_t pre_relu_2 = get_timer();
    printf("MEASUREMENT: Third (before ReLU) Convolutional Layer (in_dim=%0ldx%0ld, filter=%0ld) execution took %0ld cycles\n", 30, 30, conv_2_weight_rows, pre_relu_2);
    start_timer();
    #endif

    DEBUG_PRINTF_NOARGS("conv_model_forward: Third conv done\n");
    float_relu(28*28,
               output_conv2,
               output_relu2);

    #if defined(MEASURE)
    stop_timer();
    int64_t relu_2 = get_timer();
    printf("MEASUREMENT: ReLU (dim=%0ldx%0ld) execution took %0ld cycles (before ReLU)\n", 28, 28, relu_2);
    printf("MEASUREMENT: Third (after ReLU) Convolutional Layer (in_dim=%0ldx%0ld, filter=%0ld) execution took %0ld cycles\n", 30, 30, conv_2_weight_rows, pre_relu_2 + relu_2);
    start_timer();
    #endif

    // output layer
    float_mat_vec_product(dense_0_weight_rows,
                          dense_0_weight_columns,
                          dense_0_weight,
                          dense_0_bias,
                          output_relu2,
                          output);

    #if defined(MEASURE)
    stop_timer();
    int64_t pre_relu_3 = get_timer();
    printf("MEASUREMENT: Output Fully-Connected Layer (in_dim=%0ld, out_dim=%0ld) execution took %0ld cycles\n", dense_0_weight_columns, dense_0_weight_rows, pre_relu_3);
    printf("MEASUREMENT: Complete forward-pass took %0ld cycles\n", pre_relu_0 + relu_0 + pre_relu_1 + relu_1 + pre_relu_2 + relu_2 + padding_0 + padding_1 + padding_2 + pre_relu_3);
    start_timer();
    #endif

    #if defined(DEBUG)
    for (uint32_t i = 0; i < 10; i++) {
        char resultStr[8];
        getFloatUpTo5thDecimal(output[i], resultStr);
        printf("Formatted result: %s\n", resultStr);
    }
    #endif

    uint32_t argmax = float_argmax(10, output);

    #if defined(MEASURE)
    stop_timer();
    int64_t argmax_time = get_timer();
    printf("MEASUREMENT: Argmax computation took %0ld cycles\n", argmax_time);
    #endif

    return argmax;
}
#endif
