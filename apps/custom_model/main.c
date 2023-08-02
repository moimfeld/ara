// Author: Moritz Imfeld <moimfeld@student.ethz.ch>

// How to compile:
// ENV_DEFINES="-DUSE_VEXT=1 -DDEBUG=1 -DFC_MODEL" make bin/custom_model
// ENV_DEFINES="-DUSE_VEXT=1 -DDEBUG=1 -DFC_MODEL" make bin/custom_model.spike # for spike
//
// -D DEBUG enables debug prints
// -D FC_MODEL selects the model that should be evaluated/run
// -D USE_VECT enables vectorized kernels


#include "kernel/custom_model_forward_pass.h"
// #include "kernel/custom_model_padding.h"
// #include "kernel/custom_model_helper.h"
#include "kernel/images_labels.h"
#include <stdint.h>
#include <string.h>


#ifndef SPIKE
#include "printf.h"
#else
#include "util.h"
#include <stdio.h>
#endif

int main() {

  // float some_matrix[26*26];
  // float some_padded_matrix[30*30];
  // for (uint32_t i = 0; i < 26*26; i++) {
  //   some_matrix[i] = 100 + i;
  // }

  // float_zero_pad(some_padded_matrix, some_matrix, 26, 26, 2);

  // for (uint32_t i = 0; i < 30; i++) {
  //   for (uint32_t j = 0; j < 30; j++) {
  //       char resultStr[8];
  //       getFloatUpTo5thDecimal(some_padded_matrix[i * 30 + j], resultStr);
  //       printf("%s ", resultStr);
  //   }
  //   printf("\n\n");
  // }
  // return 0;

  #if defined(MEASURE)
  printf("Measuring enabled\n");
  #endif

  #if defined(USE_VEXT)
  printf("Using vectorized kernels\n");
  #else
  printf("Using scalar kernels\n");
  #endif

  uint8_t argmax;
  #if defined(TINY_FC_MODEL)
  printf("Evaluating TINY_FC_MODEL\n");
  argmax = tiny_fc_model_forward(simplenet_inputs[0]);
  #elif defined(FC_MODEL)
  printf("Evaluating FC_MODEL\n");
  argmax = fc_model_forward(simplenet_inputs[0]);
  #elif defined(CONV_MODEL)
  printf("Evaluating CONV_MODEL\n");
  argmax = conv_model_forward(simplenet_inputs[0]);
  #elif defined(CONV_POOL_MODEL)
  printf("Evaluating CONV_POOL_MODEL\n");
  argmax = conv_pool_model_forward(simplenet_inputs[0]);
  #elif defined(CONV_PAD_MODEL)
  printf("Evaluating CONV_PAD_MODEL\n");
  argmax = conv_pad_model_forward(simplenet_inputs[0]);
  #elif defined(RESNET)
  #error Not Implemented :(
  #else
  #warning No model defined, defaulting to TINY_FC_MODEL. Use "-D <MODEL_NAME>" to define a model (Possible models TINY_FC_MODEL, FC_MODEL, CONV_MODEL, CONV_POOL_MODEL, CONV_PAD_MODEL, RESNET)
  printf("No model defined at compile time, defaulting to TINY_FC_MODEL. Use '-D <MODEL_NAME>' to define a model (Possible models TINY_FC_MODEL, FC_MODEL, CONV_MODEL, CONV_POOL_MODEL, CONV_PAD_MODEL, RESNET)\n");
  tiny_fc_model_forward(simplenet_inputs[1]);
  #endif

  printf("Prediction: %0d\n", argmax);
  printf("Correct Label: %0d\n", simplenet_labels[0]);

  return 0;
}
