#ifndef _CUSTOM_MODEL_MACROS_H_
#define _CUSTOM_MODEL_MACROS_H_

#if defined(DEBUG)
#define DEBUG_PRINTF(formatstring, ...) \
            { printf("DEBUG: " formatstring, __VA_ARGS__); }
#else
#define DEBUG_PRINTF(formatstring, ...) {}
#endif

#endif