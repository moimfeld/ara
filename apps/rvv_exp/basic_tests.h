#ifndef BASIC_TESTS_H
#define BASIC_TESTS_H

#include "util.h"

int vector_move_load_test_64() {
    printf("\n----------------------------------\n");
    printf(  "- Vector Move Load Test - SEW=64 -\n");
    printf(  "----------------------------------\n\n");


    // Initialize array
    uint64_t vload_via_move_test[] = {3, 2, 7, 123, 9};

    // print array
    printf("Array to load:\n");
    hex_print_array(vload_via_move_test, 5);

    // load array into v_reg with vector moves
    vload_vmv_x_s(vload_via_move_test, 5, 0);

    // print loaded array
    printf("Loaded vector:\n");
    hex_print_vreg(5, 0);

    printf("\nCHECK PRINTS ABOVE TO SEE IF TEST SUCCEEDED.\n\n");

    return 0;
}

int vector_load_test_64() {
	printf("\n-----------------------------\n");
    printf(  "- Vector Load Test - SEW=64 -\n");
    printf(  "-----------------------------\n\n");

    // set and print vector config
    vsetvl(5, 64, 1);
    print_v_cfg();

    // initialize array
    uint64_t vload_test[] = {12, 47, 9, 21, 56};

    // print array
    printf("Array to load:\n");
    hex_print_array(vload_test, 5);

    // perform vector load
    asm volatile("vle64.v v0, (%0)": : "r"(vload_test));

    // print vector load result
    printf("Loaded vector:\n");
    hex_print_vreg(5, 0);

    printf("\nCHECK PRINTS ABOVE TO SEE IF TEST SUCCEEDED.\n\n");

    return 0;
}

int find_physical_address_example() {
    printf("\n---------------------------------\n");
    printf(  "- Find Physical Address Example -\n");
    printf(  "---------------------------------\n\n");

    printf("This function is an easy example of how to find the physical address\n");
    printf("of an array using special values that can serve as trigger for ILAs.\n\n");

    printf("The following trigger values can be catched on the acc_req.rs1 signal\n");
    printf("Trigger value before cva6 array load: 0x%0x\n", VMV_TRIGGER_VAL_0);
    printf("Trigger value before ara array load:  0x%0x\n", VMV_TRIGGER_VAL_1);

    // initialize test variables
    volatile uint64_t tmp_var = 0;
    uint64_t phy_addr_test[] = {1000, 2000, 3000, 4000, 5000};

    // Set vector processor configuration
    vsetvl(5, 64, 1);

    asm volatile("vmv.s.x v31, %0" : : "r"(VMV_TRIGGER_VAL_0));
    tmp_var = phy_addr_test[0];

    asm volatile("vmv.s.x v31, %0" : : "r"(VMV_TRIGGER_VAL_1));
    asm volatile("vle64.v v0, (%0)" : : "r"(phy_addr_test));

    return 0;
}


int data_copy_test_64() {
    printf("\n---------------------------\n");
    printf(  "- Data Copy Test - SEW=64 -\n");
    printf(  "---------------------------\n\n");

    // set and print vector config
    vsetvl(5, 64, 1);
    print_v_cfg();

    // initialize arrays
    uint64_t data_copy_source[] = {49153, 830624, 49374, 3809, 262254561};
    uint64_t data_copy_dest[]   = {0, 0, 0, 0, 0};

    printf("Source array:\n");
    hex_print_array(data_copy_source, 5);

    printf("Destination array:\n");
    hex_print_array(data_copy_dest, 5);

    printf("\nPerforming copy\n\n");

    asm volatile("vle64.v v0, (%0)" : : "r"(data_copy_source));
    asm volatile("vse64.v v0, (%0)" : : "r"(data_copy_dest));

    printf("Destination array after copy:\n");
    hex_print_array(data_copy_dest, 5);

    printf("\n");

    for (int i = 0; i < 5; i++) {
    	if (data_copy_source[i] != data_copy_dest[i]) {
    		printf("TEST FAILED\n");
    		printf("Elements at idx=%0d differ (source: 0x%0x; dest: 0x%0x\n", i, data_copy_source[i], data_copy_dest[i]);
    		return -1;
    	}
    }
    printf("TEST SUCCESSFUL\n");
    return 0;
}

int element_wise_multiplication_64() {
	printf("\n----------------------------------------\n");
    printf(  "- Element Wise Multiplication - SEW=64 -\n");
    printf(  "----------------------------------------\n\n");

    // set and print vector config
    vsetvl(5, 64, 1);
    print_v_cfg();

    // initialize arrays
    uint64_t elem_mul_a[] = {34, 12, 9, 61, 93};
    uint64_t elem_mul_b[] = {56, 41, 3, 84, 57};
    uint64_t elem_mul_s_res[] = {0, 0, 0, 0, 0}; // scalar result
    uint64_t elem_mul_v_res[] = {0, 0, 0, 0, 0}; // vector result

    printf("Source vector a:\n");
    hex_print_array(elem_mul_a, 5);
    printf("Source vector b;\n");
    hex_print_array(elem_mul_b, 5);
    printf("\nPerforming computations\n\n");

    // scalar execution
    for (int i = 0; i < 5; i++) {
        elem_mul_s_res[i] = elem_mul_a[i] * elem_mul_b[i];
    }

    // vector execution
    asm volatile("vle64.v v0, (%0)" : : "r"(elem_mul_a));
    asm volatile("vle64.v v1, (%0)" : : "r"(elem_mul_b));
    asm volatile("vmul.vv v2, v0, v1");
    asm volatile("vse64.v v2, (%0)" : : "r"(elem_mul_v_res));

    printf("Scalar (golden) result:\n");
    hex_print_array(elem_mul_s_res, 5);
    printf("Vector result:\n");
    hex_print_array(elem_mul_v_res, 5);

    printf("\n");

    for (int i = 0; i < 5; i++) {
    	if (elem_mul_s_res[i] != elem_mul_v_res[i]) {
    		printf("TEST FAILED\n");
    		printf("Elements at idx=%0d differ (scalar: 0x%0x; vec: 0x%0x\n", i, elem_mul_s_res[i], elem_mul_v_res[i]);
    		return -1;
    	}
    }
    printf("TEST SUCCESSFUL\n");
    return 0;
}

int counter_example() {
	printf("\n-----------------------------------\n");
    printf(  "- Performance Measurement Example -\n");
    printf(  "-----------------------------------\n\n");

	printf("Measuring the execution time of a for-loop\n\n");
	long start_cnt = get_cycle_cnt();
	volatile int tmp = 0;
	for(int i = 0; i < 1000; i++) {
		tmp += 1;
	}
	long end_cnt = get_cycle_cnt();
	printf("Start cycle count:      %10d\n", start_cnt);
	printf("End cycle count:        %10d\n", end_cnt);
	printf("Total execution cycles: %10d\n", end_cnt - start_cnt);
}

#endif // BASIC_TESTS_H
