#ifndef UTIL_H
#define UTIL_H
#include <stdio.h>

// preprocessor ToString function
#define xstr(s) str(s)
#define str(s) #s

// Define function prototypes here
#define MAX_VL            65536
#define CSR_VTYPE         0xc21
#define CSR_VL            0xc20
#define VMV_TRIGGER_VAL_0 0xc1a0
#define VMV_TRIGGER_VAL_1 0x210
#define CSR_CYCLE         0xC00


void help(){
	printf("This program can be used to verify the functionality of Ara in Cheshire\n");
	printf("\n");
	printf("\n");
	printf("How to control the program:\n");
	printf("\n");
	printf("\n");
	return;
}

long __attribute__((always_inline)) get_cycle_cnt() {
	long cycle_cnt;
	asm volatile("csrr %0, " xstr(CSR_CYCLE) : "=r"(cycle_cnt));
	return cycle_cnt;
}

long __attribute__((always_inline)) get_vl() {
	long vl;
	asm volatile("csrr %0, " xstr(CSR_VL) : "=r"(vl));
	return vl;
}

long __attribute__((always_inline)) get_vtype() {
	long vtype;
	asm volatile("csrr %0, " xstr(CSR_VTYPE) : "=r"(vtype));
	return vtype;
}

long __attribute__((always_inline)) set_vl_and_vtype(long avl, long vtype) {
	long vl;
	asm volatile("vsetvl %0, %1, %2" : "=r"(vl) : "r"(avl), "r"(vtype));
	return vl;
}

long __attribute__((always_inline)) get_sew() {
	long vtype = get_vtype();
	long shift = 3;
	long mask  = 0x7;
	long sew = (vtype >> shift) & mask;
	// TODO must apply filter to get sew from vtype
	switch (sew) {
		case 0b000: return  8;
		case 0b001: return 16;
		case 0b010: return 32;
		case 0b011: return 64;
		default: return -1;
	}
}

long __attribute__((always_inline)) get_lmul() {
	long vtype = get_vtype();
	long mask  = 0x7;
	long lmul = vtype & mask;
	// TODO must apply filter to get lmul from vtype
	switch (lmul) {
		case 0b000: return 1;
		case 0b001: return 2;
		case 0b010: return 4;
		case 0b011: return 8;
		default: return -1;
	}
}


int find_vlmax() {
	// get current vtype and vl
	long vl = get_vl();
	long vtype = get_vtype();

	// get vlmax
	int vlmax;
	asm volatile("vsetvli %0, %1, e8, m8, ta, ma" : "=r"(vlmax) : "r"(MAX_VL));

	// restore vtype and vl
	set_vl_and_vtype(vl, vtype);
	return vlmax;
}

void print_v_cfg() {
	printf("Current Vector Register Cofiguration:\n");
	printf("VL:   %5d\n", get_vl());
	printf("SEW:  %5d\n", get_sew());
	printf("LMUL: %5d\n", get_lmul());
	printf("\n");
}

long vsetvl(int avl, int sew, int lmul) {

	int sew_enc;
	int lmul_enc;

	// check sew
	switch (sew) {
	    case 8:  sew_enc = 0b000; break;
	    case 16: sew_enc = 0b001; break;
	    case 32: sew_enc = 0b010; break;
	    case 64: sew_enc = 0b011; break;
	    default:
	        printf("[ERROR] Invalid sew (sew=%0d). Valid options are 8, 16, 32, 64.\n", sew);
	        return -1;
	}

	// check lmul
	switch (lmul) {
	    case 1: lmul_enc = 0b000; break;
	    case 2: lmul_enc = 0b001; break;
	    case 4: lmul_enc = 0b010; break;
	    case 8: lmul_enc = 0b011; break;
	    default:
	        printf("[ERROR] Invalid lmul (lmul=%0d). Valid options are 1, 2, 4, 8.\n", lmul);
	        return -1;
	}

	// build vtype
	long vtype = 0;
	vtype = (vtype +        1 ) << 1; // set vector mask agnostic
	vtype = (vtype +        1 ) << 3; // set vector tail agnostic
	vtype = (vtype +  sew_enc ) << 3; // set sew
	vtype = (vtype + lmul_enc );	  // set lmul

	// set vl and vtype
	long vl = set_vl_and_vtype(avl, vtype);
	if (vl != avl) {
		printf("[WARNING] AVL (=%0d) too large. Could only set it to VL=%0d\n", avl, vl);
	}
	return vl;
}

// can only operate on v0 to v3
// will use v4 and v5 as temporary registers
// for now set sew to 64
// for now set lmul to 1
int vload_vmv_x_s(uint64_t * array, int len, int v_reg) {
	long  sew = 64;
	long lmul = 1;
	long vl = vsetvl(len, sew, lmul);

	if (vl != len) {
		printf("[WARNING] len (=%0d) too long. Could not fit whole array into vector register. (VL=%0d)\n", len, vl);
	}

	// load data into vector register
	asm volatile("vmv.s.x v4, %0" : : "r"(array[vl-1]));
	for (int i = vl-1; i > 0; i--) {
		asm volatile("vslide1up.vx v5, v4, %0" : : "r"(array[i-1]));
		asm volatile("vmv.v.v v4, v5");
	}

	switch (v_reg) {
		case 0: asm volatile("vmv.v.v v0, v4"); break;
		case 1: asm volatile("vmv.v.v v1, v4"); break;
		case 2: asm volatile("vmv.v.v v2, v4"); break;
		case 3: asm volatile("vmv.v.v v3, v4"); break;
		default:
			printf("[ERROR] v_reg (=%0d) must be between 0 and 4\n", v_reg);
			return -1;
	}
	return 0;
}

// Function prints vector register content without using vector load or store instructions
// - can only print v0 to v3
// - will use v4 and v5 as temporary registers
// - for now set sew to 64
// - for now set lmul to 1
int print_vreg(int len, int v_reg, int prt_hex) {
	long sew = 64;
	long lmul = 1;
	long vl = vsetvl(len, sew, lmul);

	if (vl != len) {
		printf("[WARNING] len (=%0d) too long. Could not fit whole array into vector register. (VL=%0d)\n", len, vl);
	}

	// first move vector register from v_reg to v4
	switch (v_reg) {
		case 0: asm volatile("vmv.v.v v4, v0"); break;
		case 1: asm volatile("vmv.v.v v4, v1"); break;
		case 2: asm volatile("vmv.v.v v4, v2"); break;
		case 3: asm volatile("vmv.v.v v4, v3"); break;
		default:
			printf("[ERROR] v_reg (=%0d) must be between 0 and 4\n", v_reg);
			return -1;
	}

	// perform printing
	uint64_t element;
	printf("v%0d=[ ", v_reg);
	for (int i = 0; i < vl; i++) {
		asm volatile("vmv.x.s %0, v4" : "=r"(element));
		if (prt_hex == 1) {
			printf("0x%0x", element);
		} else {
			printf("%0d", element);
		}
		if (i != (vl-1)) {
			printf(", ");
		}
		asm volatile("vslidedown.vx v5, v4, %0" : : "r"(1));
		asm volatile("vmv.v.v v4, v5");
	}
	printf("]\n");
}

int dec_print_vreg(int len, int v_reg) {
	return print_vreg(len, v_reg, 0);
}

int hex_print_vreg(int len, int v_reg) {
	return print_vreg(len, v_reg, 1);
}

// only accepts uint64_t array
int print_array(uint64_t * array, int len, int prt_hex) {
	printf("array=[ ");
	for (int i = 0; i < len; i++) {
		if (prt_hex == 1) {
			printf("0x%0x", array[i]);
		} else {
			printf("%0d", array[i]);
		}
		
		if (i != (len-1)) {
			printf(", ");
		}
	}
	printf("]\n");
}

int hex_print_array(uint64_t * array, int len) {
	return print_array(array, len, 1);
}

int dec_print_array(uint64_t * array, int len) {
	return print_array(array, len, 0);
}

#endif /* UTIL_H */
