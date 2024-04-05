// Copyright 2021 ETH Zurich and University of Bologna.
// Solderpad Hardware License, Version 0.51, see LICENSE for details.
// SPDX-License-Identifier: SHL-0.51
//
// Author: Matheus Cavalcante <matheusd@iis.ee.ethz.ch>
//         Basile Bougenot <bbougenot@student.ethz.ch>

#include "vector_macros.h"

void TEST_CASE1() {
  VSET(4, e64, m1);
  VLOAD_64(v4, 1, 2, 3, 4);
  VLOAD_64(v0, 12, 0, 0, 0);
  VCLEAR(v2);
  __asm__ volatile("vcompress.vm v2, v4, v0");
  VCMP_U64(1, v2, 3, 4, 0, 0);
}

void TEST_CASE2() {
  VSET(16, e8, m1);
  VLOAD_8(v4, 1, 2, 3, 4,
              5, 6, 7, 8,
              9, 10, 11, 12,
              13, 14, 15, 16);
  VLOAD_8(v0, 0b10111100, 0b10111111, 0, 0, 0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0);
  VCLEAR(v2);
  __asm__ volatile("vcompress.vm v2, v4, v0");
  VCMP_U8(2, v2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 16, 0, 0, 0, 0);
}


int main(void) {
  INIT_CHECK();
  enable_vec();
  enable_fp();
  TEST_CASE1();
  TEST_CASE2();
  EXIT_CHECK();
}
