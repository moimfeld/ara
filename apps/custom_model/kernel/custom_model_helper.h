#ifndef _CUSTOM_MODEL_HELPER_H_
#define _CUSTOM_MODEL_HELPER_H_

#include <stdio.h>
void getFloatUpTo5thDecimal(float num, char resultStr[8]) {
    // Convert float to integer by multiplying by 100000
    int64_t integerPart = (int64_t)num;
    int64_t decimalPart = (int64_t)((num - integerPart) * 100000);

    if (decimalPart < 0) {
        decimalPart = decimalPart * (-1);
    }

    // Create the formatted string (up to 4th decimal place)
    sprintf(resultStr, "%d.%05d", integerPart, decimalPart);
}

#endif