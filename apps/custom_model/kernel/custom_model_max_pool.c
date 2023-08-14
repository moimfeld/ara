
#include "custom_model_max_pool.h"

#define block_size_2x2 2
#define TILE_SIZE 256

// *o : tensor convolution output pointer k x C x R (dimmensions depends on the filter size and stride)
// *i : input tensor pointer 1 x W x C x R
// H_in  : number of input Rows
// W_in  : number of input Column
// C_in  : channels of the input tensor
// F  : size of the filter

void fmax_pool_vec_1xC_2x2(float *o, float *i, int64_t R, int64_t C, int64_t W, int64_t F, int64_t stride) {
	#if defined(USE_VEXT)
	int64_t ld = stride << 2;
	
	// vsetvli C (in order to load everything)
	asm volatile("vsetvli zero, %0, e32, m4, ta, ma" ::"r"(TILE_SIZE >> 1)); // we only load one of two values 
	
	for (int c = 0 ; c < C ; c += TILE_SIZE)
	{
		if(c > C - TILE_SIZE)
			asm volatile("vsetvli zero, %0, e32, m4, ta, ma" ::"r"((C % TILE_SIZE) >> 1)); // we only load one of two values 

		float * i_ = i + c; 			// input pointer realtive to the tile (constant throughout the tile)
		float * o_ = o + (c >> 1);			// output pointer relative to the tile	
		
		// start VCD_DUMP
		#if defined(VCD_DUMP)
		event_trigger = +1;
		#endif

		//channel and height loop are fused to avoid branches
		for (int r = 0 ; r < R * W ; r += block_size_2x2){
			
		 	// Load F row of size C
		 	asm volatile("vlse32.v v16, (%0), %1" : "+&r"(i_) : "r"(ld));
		 	i_ += 1;
		 	asm volatile("vlse32.v v20, (%0), %1" : "+&r"(i_) : "r"(ld));
		 	i_ += C - 1;
		 	// next line
		 	asm volatile("vlse32.v v24, (%0), %1" : "+&r"(i_) : "r"(ld));
		 	i_ += 1;
		 	asm volatile("vlse32.v v28, (%0), %1" : "+&r"(i_) : "r"(ld));
		 	i_ += C - 1;
				
			// max function between the F rows
		 	asm volatile("vfmax.vv v4,  v20,  v16");
		 	asm volatile("vfmax.vv v8,  v24,  v28");
		 	asm volatile("vfmax.vv v0,  v4,  v8");
		 		
		 	// store
			asm volatile("vse32.v v0, (%0)" : "+&r"(o_));
			o_ += C >> 1;
	  	 
  		}

		// stop VCD_DUMP
		#if defined(VCD_DUMP)
		event_trigger = -1;
		#endif
  	}
	#else
		for (uint32_t k = 0; k < R/2; k += 1) {
			for (uint32_t j = 0; j < C/2; j += 1) {
				o[k*(R/2) + j] = i[2*k*R + 2*j];
				if (o[k*(R/2) + j] < i[2*k*R + R + 2*j]) {
					o[k*(R/2) + j] = i[2*k*R + R + 2*j];
				}
				if (o[k*(R/2) + j] < i[2*k*R + 2*j+1]) {
					o[k*(R/2) + j] = i[2*k*R + 2*j+1];
				}
				if (o[k*(R/2) + j] < i[2*k*R + R + 2*j+1]) {
					o[k*(R/2) + j] = i[2*k*R + R + 2*j+1];
				}
			}
		}
	#endif

}