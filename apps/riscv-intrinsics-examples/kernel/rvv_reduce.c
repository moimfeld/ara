#include "common.h"
#include "rvv_reduce.h"
#include <riscv_vector.h>


// accumulate and reduce
void reduce_golden(double *a, double *b, double *result_sum,
                     int *result_count, int n) {
  int count = 0;
  double s = 0;
  for (int i = 0; i < n; ++i) {
    if (a[i] != 42.0) {
      s += a[i] * b[i];
      count++;
    }
  }

  *result_sum = s;
  *result_count = count;
}
/*
v0 - mask
v1 - vec_zero
v2 - vec_s
v3 - vec_a
v4 - vec_b
v5 - vec_sum
*/
void reduce_vec(double *a, double *b, double *result_sum, int *result_count,
                int n) {
  int count = 0;
  int vcpop_res;

  // set vlmax and initialize variables
  size_t vlmax = 20000; // = __riscv_vsetvlmax_e64m1();
  asm volatile("vsetvli %0, %1, e64, m1, ta, ma" : "=r"(vlmax) : "r"(vlmax));
  // vfloat64m1_t vec_zero = __riscv_vfmv_v_f_f64m1(0, vlmax); vec_zero is saved in v1
  // vfloat64m1_t vec_s = __riscv_vfmv_v_f_f64m1(0, vlmax);    vec_s    is saved in v2
  asm volatile("vfmv.v.f v1, %0" : : "f"(0.0f));
  asm volatile("vfmv.v.f v2, %0" : : "f"(0.0f));
  for (size_t vl; n > 0; n -= vl, a += vl, b += vl) {
    // vl = __riscv_vsetvl_e64m1(n);
    asm volatile("vsetvli %0, %1, e64, m1, ta, ma" : "=r"(vl) : "r"(n));

    // vfloat64m1_t vec_a = __riscv_vle64_v_f64m1(a, vl); vec_a is saved in v3
    // vfloat64m1_t vec_b = __riscv_vle64_v_f64m1(b, vl); vec_b is saved in v4
    asm volatile("vle64.v v3, (%0)" : : "r"(a));
    asm volatile("vle64.v v4, (%0)" : : "r"(b));

    // vbool64_t mask = __riscv_vmfne_vf_f64m1_b64(vec_a, 42, vl); mask is saved in v0
    double limit = 42.0f;
    asm volatile("vmfne.vf v0, v3, %0" : : "f"(limit));


    // vec_s = __riscv_vfmacc_vv_f64m1_tumu(mask, vec_s, vec_a, vec_b, vl);
    asm volatile("vfmacc.vv v2, v3, v4, v0.t");
    // count = count + __riscv_vcpop_m_b64(mask, vl);
    asm volatile("vcpop.m %0, v0, v0.t" : "=r"(vcpop_res));
    count = count + vcpop_res;
  }
  // vfloat64m1_t vec_sum; vec_sum is saved in v5
  // vec_sum = __riscv_vfredusum_vs_f64m1_f64m1(vec_s, vec_zero, vlmax);
  asm volatile("vsetvli %0, %1, e64, m1, ta, ma" : "=r"(vlmax) : "r"(vlmax));
  asm volatile("vfredusum.vs v5, v2, v1, v0.t");

  // double sum = __riscv_vfmv_f_s_f64m1_f64(vec_sum);
  double sum;
  asm volatile("vfmv.f.s %0, v5" : "=f"(sum)); // MOIMFELD: Note - vmf.f.s not working for now

  *result_sum = sum;
  *result_count = count;
}

void reduce_main() {
  const int N = 31;
  // uint32_t seed = 0xdeadbeef;
  // srand(seed);

  // data gen
  printf("Starting initializing Arrays\n");
  double A[N], B[N];
  // gen_rand_1d(A, N);
  // gen_rand_1d(B, N);
  for (int i = 0; i < N; i++) {
    A[i] = i * 1.0f + 20.0f;
    B[i] = i * 1.0f + 10.0f;
  }
  printf("Finished initializing Arrays\n");


  // compute
  double golden_sum, actual_sum;
  int golden_count, actual_count;
  printf("Starting golden reduction\n");
  reduce_golden(A, B, &golden_sum, &golden_count, N);
  printf("Finished golden reduction\n");
  printf("Starting vector reduction\n");
  reduce_vec(A, B, &actual_sum, &actual_count, N);
  printf("Finished vector reduction\n");

  // compare
  if (golden_count == actual_count) { // ADD: golden_sum - actual_sum < 1e-6) && 
    printf("reduction pass\n", golden_count, actual_count);
  } else {
    // double difference = golden_sum - actual_sum;
    printf("reduction fail (golden_sum = %x, actual_sum = %x, golden_count = %0d, actual_count = %0d)\n", golden_sum, actual_sum, golden_count, actual_count);
  }
}
