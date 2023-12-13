#ifndef _RVV_REDUCE_H_
#define _RVV_REDUCE_H_

void reduce_main();

void reduce_vec(double *a, double *b, double *result_sum, int *result_count, int n);

void reduce_golden(double *a, double *b, double *result_sum, int *result_count, int n);

#endif