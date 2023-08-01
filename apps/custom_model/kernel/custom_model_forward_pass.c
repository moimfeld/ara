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

    // First Layer
    start_timer();
    float_mat_vec_product(dense_0_weight_rows,
                            dense_0_weight_columns,
                            dense_0_weight,
                            dense_0_bias,
                            input,
                            dense_0_output);
    float_relu(dense_0_weight_rows,
                dense_0_output,
                relu_0_output);
    stop_timer();
    int64_t runtime = get_timer();
    printf("First Layer execution took %0ld cycles.\n", runtime);

    // Second Layer
    start_timer();
    float_mat_vec_product(dense_1_weight_rows,
                            dense_1_weight_columns,
                            dense_1_weight,
                            dense_1_bias,
                            relu_0_output,
                            dense_1_output);
    float_relu(dense_1_weight_rows,
                dense_1_output,
                relu_1_output);
    stop_timer();
    runtime = get_timer();
    printf("Second layer execution took %0ld cycles.\n", runtime);

    // Third Layer
    start_timer();
    float_mat_vec_product(dense_2_weight_rows,
                            dense_2_weight_columns,
                            dense_2_weight,
                            dense_2_bias,
                            relu_1_output,
                            dense_2_output);
    float_relu(dense_2_weight_rows,
                dense_2_output,
                relu_2_output);
    stop_timer();
    runtime = get_timer();
    printf("Third layer execution took %0ld cycles.\n", runtime);

    // Output Layer
    start_timer();
    float_mat_vec_product(dense_3_weight_rows,
                            dense_3_weight_columns,
                            dense_3_weight,
                            dense_3_bias,
                            relu_2_output,
                            dense_3_output);
    stop_timer();
    runtime = get_timer();
    printf("Last layer execution took %0ld cycles.\n", runtime);

    uint32_t argmax = float_argmax(10, dense_3_output);
    return argmax;
}
#endif

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
    start_timer();
    float_mat_vec_product(dense_0_weight_rows,
                            dense_0_weight_columns,
                            dense_0_weight,
                            dense_0_bias,
                            input,
                            dense_0_output);
    float_relu(dense_0_weight_rows,
                dense_0_output,
                relu_0_output);
    stop_timer();
    int64_t runtime = get_timer();
    printf("Last layer execution took %0ld cycles.\n", runtime);

    // Second Layer
    start_timer();
    float_mat_vec_product(dense_1_weight_rows,
                            dense_1_weight_columns,
                            dense_1_weight,
                            dense_1_bias,
                            relu_0_output,
                            dense_1_output);
    float_relu(dense_1_weight_rows,
                dense_1_output,
                relu_1_output);
    stop_timer();
    runtime = get_timer();
    printf("Last layer execution took %0ld cycles.\n", runtime);

    // Third Layer
    start_timer();
    float_mat_vec_product(dense_2_weight_rows,
                            dense_2_weight_columns,
                            dense_2_weight,
                            dense_2_bias,
                            relu_1_output,
                            dense_2_output);
    float_relu(dense_2_weight_rows,
                dense_2_output,
                relu_2_output);
    stop_timer();
    runtime = get_timer();
    printf("Last layer execution took %0ld cycles.\n", runtime);

    // Output Layer
    start_timer();
    float_mat_vec_product(dense_3_weight_rows,
                            dense_3_weight_columns,
                            dense_3_weight,
                            dense_3_bias,
                            relu_2_output,
                            dense_3_output);
    stop_timer();
    runtime = get_timer();
    printf("Last layer execution took %0ld cycles.\n", runtime);

    uint32_t argmax = float_argmax(10, dense_3_output);
    return argmax;
}
#endif

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

    conv_2d(28,
            28,
            conv_0_weight_rows,
            conv_0_weight,
            conv_0_bias,
            input,
            output_conv0);

    DEBUG_PRINTF_NOARGS("conv_model_forward: First conv done\n");
    float_relu(26*26,
               output_conv0,
               output_relu0);
    conv_2d(26,
            26,
            conv_1_weight_rows,
            conv_1_weight,
            conv_1_bias,
            output_relu0,
            output_conv1);
    DEBUG_PRINTF_NOARGS("conv_model_forward: Second conv done\n");
    float_relu(24*24,
               output_conv1,
               output_relu1);
    conv_2d(24,
            24,
            conv_2_weight_rows,
            conv_2_weight,
            conv_2_bias,
            output_relu1,
            output_conv2);
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
    float_mat_vec_product(dense_0_weight_rows,
                          dense_0_weight_columns,
                          dense_0_weight,
                          dense_0_bias,
                          output_relu2,
                          output);

    #if defined(DEBUG)
    for (uint32_t i = 0; i < 10; i++) {
        char resultStr[8];
        getFloatUpTo5thDecimal(output[i], resultStr);
        printf("Formatted result: %s\n", resultStr);
    }
    #endif

    uint32_t argmax = float_argmax(10, output);
    return argmax;
}
#endif

#if defined(CONV_POOL_MODEL)
uint8_t conv_pool_model_forward(float * input)
{
    float output_conv0[26*26];
    float output_relu0[26*26];
    float output_conv1[24*24];
    float output_relu1[24*24];
    float output_maxPool[12*12];
    float output[10];

    conv_2d(28,
            28,
            conv_0_weight_rows,
            conv_0_weight,
            conv_0_bias,
            input,
            output_conv0);

    DEBUG_PRINTF_NOARGS("conv_model_forward: First conv done\n");
    float_relu(26*26,
               output_conv0,
               output_relu0);
    conv_2d(26,
            26,
            conv_1_weight_rows,
            conv_1_weight,
            conv_1_bias,
            output_relu0,
            output_conv1);
    DEBUG_PRINTF_NOARGS("conv_model_forward: Second conv done\n");
    float_relu(24*24,
               output_conv1,
               output_relu1);

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


    float_mat_vec_product(dense_0_weight_rows,
                          dense_0_weight_columns,
                          dense_0_weight,
                          dense_0_bias,
                          output_maxPool,
                          output);

    #if defined(DEBUG)
    for (uint32_t i = 0; i < 10; i++) {
        char resultStr[8];
        getFloatUpTo5thDecimal(output[i], resultStr);
        printf("Formatted result: %s\n", resultStr);
    }
    #endif

    uint32_t argmax = float_argmax(10, output);
    return argmax;
}
#endif

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


    // output, input, i_rows, i_columns, padding
    float_zero_pad(output_pad0, input, 28, 28, 1);
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
    

    DEBUG_PRINTF_NOARGS("conv_model_forward: First conv done\n");
    float_relu(28*28,
               output_conv0,
               output_relu0);


    float_zero_pad(output_pad1, output_relu0, 28, 28, 1);
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
    DEBUG_PRINTF_NOARGS("conv_model_forward: Second conv done\n");
    float_relu(28*28,
               output_conv1,
               output_relu1);

    float_zero_pad(output_pad2, output_relu1, 28, 28, 1);
    conv_2d(30,
            30,
            conv_2_weight_rows,
            conv_2_weight,
            conv_2_bias,
            output_relu1,
            output_conv2);
    DEBUG_PRINTF_NOARGS("conv_model_forward: Third conv done\n");
    float_relu(28*28,
               output_conv2,
               output_relu2);



    float_mat_vec_product(dense_0_weight_rows,
                          dense_0_weight_columns,
                          dense_0_weight,
                          dense_0_bias,
                          output_relu2,
                          output);

    #if defined(DEBUG)
    for (uint32_t i = 0; i < 10; i++) {
        char resultStr[8];
        getFloatUpTo5thDecimal(output[i], resultStr);
        printf("Formatted result: %s\n", resultStr);
    }
    #endif

    uint32_t argmax = float_argmax(10, output);
    return argmax;
}
#endif
