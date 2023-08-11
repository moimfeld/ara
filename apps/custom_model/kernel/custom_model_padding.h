#ifndef _CUSTOM_MODEL_PADDING_H_
#define _CUSTOM_MODEL_PADDING_H_

#include <stdint.h>


extern int64_t event_trigger;

// idea: pad at top and bottom with empty vectors
//       slide vectors to pad left side of vectors
//       scalar instructions to pad right side
void float_zero_pad(float *output, float *input, uint64_t i_rows, uint64_t i_column, uint64_t pad);


#endif