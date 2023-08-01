#ifndef _CUSTOM_MODEL_CONV2D_H_
#define _CUSTOM_MODEL_CONV2D_H_

#include <stdint.h>

#define TILE_SIZE 256
#define TILE_SIZE_OUT (TILE_SIZE - F + 1)

void fconv2d_tensor32_vec_6xC_3x3(float *o, float *i, float *f, int64_t R, int64_t C, int64_t W, int64_t F);

#endif