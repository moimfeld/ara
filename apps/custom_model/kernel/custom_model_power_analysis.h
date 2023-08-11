#ifndef _CUSTOM_MODEL_KERNEL_BM_H_
#define _CUSTOM_MODEL_KERNEL_BM_H_

#include "stdint.h"

// KERNEL PARAMETERS
#define MAT_VEC_INPUT_SIZE  768
#define MAT_VEC_OUTPUT_SIZE 10

#define CONV_FILTER_SIZE 3
#define CONV_INPUT_SIZE 30
#define CONV_INPUT_CHANNEL 1

#define PAD_PADDING 1
#define PAD_INPUT_SIZE 28

#define POOL_SIZE 2
#define POOL_INPUT_SIZE 28
#define POOL_INPUT_CHANNEL 1
#define POOL_WINDOW_SIZE 2
#define POOL_STRIDE 2

#define RELU_INPUT_SIZE 28*28

void power_analysis();

#endif