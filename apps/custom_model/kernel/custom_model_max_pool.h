#ifndef _CUSTOM_MODEL_MAX_POOL_H_
#define _CUSTOM_MODEL_MAX_POOL_H_

#include <stdint.h>

extern int64_t event_trigger;

void fmax_pool_vec_1xC_2x2(float *o, float *i, int64_t R, int64_t C, int64_t W, int64_t F, int64_t stride);

#endif