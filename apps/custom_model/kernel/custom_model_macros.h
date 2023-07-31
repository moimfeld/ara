#ifndef _CUSTOM_MODEL_MACROS_H_
#define _CUSTOM_MODEL_MACROS_H_

#if defined(DEBUG)
#define DEBUG_PRINTF(formatstring, ...) \
        printf("DEBUG: " formatstring, __VA_ARGS__);

#define DEBUG_PRINTF_NOARGS(formatstring) \
        printf("DEBUG: " formatstring);
#else
#define DEBUG_PRINTF(formatstring, ...) {}
#define DEBUG_PRINTF_NOARGS(formatstring) {}
#endif

#endif