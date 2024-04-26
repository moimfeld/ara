#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#include "util.h"
#include "basic_tests.h"

int main(int argc, char *argv[]){
    // #ifdef SPIKE
    // argc = 5;
    // char *custom_argv[] = {"filepath", "test_vsetvl", "4", "8", "1"};
    // argv = custom_argv;
    // #endif

    // if (argc != 5) {
    //     printf("argc != 5 (argc=%0d)\n", argc);
    //     return 0;
    // }

    // asm volatile("vsetvli zero, %0, e8, m8, ta, ma" : : "r"(102));

    // find platform vlmax
    int vlmax = find_vlmax();
    printf("vlmax=%0d\n", vlmax);

    // long vl = get_vl();
    // long vtype = get_vtype();

    // printf("vl=%0ld\n", vl);
    // printf("vtype=0x%0x\n", vtype);


    // long  avl = atol(argv[2]);
    // long elen = atol(argv[3]);
    // long lmul = atol(argv[4]);


    // vsetvl(avl, elen, lmul);
    
    // vector move load test
    vector_move_load_test_64();

    // vector load test
    vector_load_test_64();

    // How to find physical address of an array example
    find_physical_address_example();

    // Data copy test
    data_copy_test_64();

    // Elementwise multiplication
    element_wise_multiplication_64();

    // Performance Measurement Example
    counter_example();


    return 0;
}
