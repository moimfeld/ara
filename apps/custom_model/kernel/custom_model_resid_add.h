#ifndef _CUSTOM_MODEL_RESID_ADD_H_
#define _CUSTOM_MODEL_RESID_ADD_H_

#include "stdint.h"

void float_resid_add(float *output, float *input, float * residual, uint32_t len);

#endif