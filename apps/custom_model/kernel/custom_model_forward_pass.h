#ifndef _CUSTOM_MODEL_FORWARD_PASS_H_
#define _CUSTOM_MODEL_FORWARD_PASS_H_

#include <stdint.h>
#if defined(USE_VEXT)
#include <riscv_vector.h>
#endif

uint8_t tiny_fc_model_forward(float * input);
uint8_t fc_model_forward(float * input);
uint8_t conv_model_forward(float * input);
uint8_t conv_pool_model_forward(float * input);



#endif