#include "custom_model_conv2d.h"

#define next_plane_(a) ((R - a + 1)*C) << 2

#define block_size_3x3 6


// R is Rows (H_in)
// C is Column (W_in)
// W is Channels (C_in)
// F is Filter (F)

void fconv2d_tensor32_vec_6xC_3x3(float *o, float *i, float *f, int64_t R, int64_t C, int64_t W, int64_t F)
{
	
int64_t const ldo = (C - 2) << 2;
int64_t const ldi = C << 2;

int64_t vlen;

int64_t const last_group = (R - F + 1) % block_size_3x3;

for (int c = 0 ; c < (C - 2) ; c += TILE_SIZE_OUT) // IF CONVOLUTION NEED TO BE TILED (C > TILE_SIZE)
{

	
	float *f_ = f;
	float *i_ = i + c; 									// input pointer realtive to the tile (constant throughout the tile)
	float *o_ = o + c;									// output pointer relative to the tile		


	
	if(c > C - TILE_SIZE) 	// if we are at the right border of the input
		vlen = C % TILE_SIZE_OUT;		 	// we set the vector length to fit the last inputs
	else
		vlen = TILE_SIZE;						// else we go full length
	
	float * i__ = i_;							// inside tile pointer (change at each load)
	
	int64_t vlen_out = vlen - 2;
	  	  	


	asm volatile("vsetvli zero, %0, e32, m2, ta, ma" ::"r"(vlen));

	asm volatile("vle32.v v12, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
	asm volatile("vle32.v v14, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(next_plane_(2)));

	asm volatile("vfmul.vf v0,  v12, %0" :: "f"(f_[0]));
	asm volatile("vfmul.vf v2,  v14, %0" :: "f"(f_[0]));
	
	asm volatile("vslidedown.vi v12, v12, 1");

	asm volatile("vfmacc.vf v0,  %0, v14" ::"f"(f_[3]));

	asm volatile("vslidedown.vi v14, v14, 1");

	asm volatile("vfmacc.vf v0,  %0, v12" ::"f"(f_[1]));
	asm volatile("vfmacc.vf v2,  %0, v14" ::"f"(f_[1]));

	asm volatile("vslidedown.vi v12, v12, 1");

	asm volatile("vfmacc.vf v0,  %0, v14" ::"f"(f_[4]));

	asm volatile("vslidedown.vi v14, v14, 1");

	asm volatile("vfmacc.vf v0,  %0, v12" ::"f"(f_[2]));
	asm volatile("vfmacc.vf v2,  %0, v14" ::"f"(f_[2]));

	asm volatile("vfmacc.vf v0,  %0, v14" ::"f"(f_[5]));


	for(int ch = 1 ; ch < W ; ch ++){

		f_ += 9;

		asm volatile("vle32.v v12, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
		asm volatile("vle32.v v14, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(next_plane_(2)));

		asm volatile("vfmacc.vf v0,  %0, v12" ::"f"(f_[0]));
		asm volatile("vfmacc.vf v2,  %0, v14" ::"f"(f_[0]));

		asm volatile("vslidedown.vi v12, v12, 1");

		asm volatile("vfmacc.vf v0,  %0, v14" ::"f"(f_[3]));

		asm volatile("vslidedown.vi v14, v14, 1");
		
		asm volatile("vfmacc.vf v0,  %0, v12" ::"f"(f_[1]));
		asm volatile("vfmacc.vf v2,  %0, v14" ::"f"(f_[1]));

		asm volatile("vslidedown.vi v12, v12, 1");

		asm volatile("vfmacc.vf v0,  %0, v14" ::"f"(f_[4]));

		asm volatile("vslidedown.vi v14, v14, 1");

		asm volatile("vfmacc.vf v0,  %0, v12" ::"f"(f_[2]));
		asm volatile("vfmacc.vf v2,  %0, v14" ::"f"(f_[2]));

		asm volatile("vfmacc.vf v0,  %0, v14" ::"f"(f_[5]));

		}


	i__ = i_ + 2 * C;
	f_ = f;

	for (int r = 2 + block_size_3x3; r < R ; r += block_size_3x3)
	{

		asm volatile("vsetvli zero, %0, e32, m2, ta, ma" ::"r"(vlen));

		asm volatile("vle32.v v16, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
		asm volatile("vle32.v v18, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));

		asm volatile("vfmul.vf v4,  v16, %0" :: "f"(f_[0]));
		asm volatile("vfmacc.vf v2,  %0, v16" ::"f"(f_[3]));
		asm volatile("vfmacc.vf v0,  %0, v16" ::"f"(f_[6]));

		asm volatile("vslidedown.vi v16, v16, 1");

		asm volatile("vle32.v v20, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));

		asm volatile("vfmul.vf v6,  v18, %0" :: "f"(f_[0]));
		asm volatile("vfmacc.vf v4,  %0, v18" ::"f"(f_[3]));
		asm volatile("vfmacc.vf v2,  %0, v18" ::"f"(f_[6]));

		asm volatile("vslidedown.vi v18, v18, 1");

		asm volatile("vle32.v v22, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));

		asm volatile("vfmul.vf v8,  v20, %0" :: "f"(f_[0]));
		asm volatile("vfmacc.vf v6,  %0, v20" ::"f"(f_[3]));
		asm volatile("vfmacc.vf v4,  %0, v20" ::"f"(f_[6]));

		asm volatile("vslidedown.vi v20, v20, 1");

		asm volatile("vle32.v v24, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));

		asm volatile("vfmul.vf v10,  v22, %0" :: "f"(f_[0]));
		asm volatile("vfmacc.vf v8,  %0, v22" ::"f"(f_[3]));
		asm volatile("vfmacc.vf v6,  %0, v22" ::"f"(f_[6]));

		asm volatile("vslidedown.vi v22, v22, 1");

		asm volatile("vle32.v v26, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(next_plane_(6)));

		asm volatile("vfmul.vf v12,  v24, %0" :: "f"(f_[0]));
		asm volatile("vfmacc.vf v10,  %0, v24" ::"f"(f_[3]));
		asm volatile("vfmacc.vf v8,  %0, v24" ::"f"(f_[6]));

		asm volatile("vslidedown.vi v24, v24, 1");

		asm volatile("vfmul.vf v14,  v26, %0" :: "f"(f_[0]));
		asm volatile("vfmacc.vf v12,  %0, v26" ::"f"(f_[3]));
		asm volatile("vfmacc.vf v10,  %0, v26" ::"f"(f_[6]));

		asm volatile("vslidedown.vi v26, v26, 1");

		asm volatile("vfmacc.vf v4,  %0, v16" ::"f"(f_[1]));
		asm volatile("vfmacc.vf v2,  %0, v16" ::"f"(f_[4]));
		asm volatile("vfmacc.vf v0,  %0, v16" ::"f"(f_[7]));

		asm volatile("vslidedown.vi v16, v16, 1");

		asm volatile("vfmacc.vf v6,  %0, v18" ::"f"(f_[1]));
		asm volatile("vfmacc.vf v4,  %0, v18" ::"f"(f_[4]));
		asm volatile("vfmacc.vf v2,  %0, v18" ::"f"(f_[7]));

		asm volatile("vslidedown.vi v18, v18, 1");

		asm volatile("vfmacc.vf v8,  %0, v20" ::"f"(f_[1]));
		asm volatile("vfmacc.vf v6,  %0, v20" ::"f"(f_[4]));
		asm volatile("vfmacc.vf v4,  %0, v20" ::"f"(f_[7]));

		asm volatile("vslidedown.vi v20, v20, 1");

		asm volatile("vfmacc.vf v10,  %0, v22" ::"f"(f_[1]));
		asm volatile("vfmacc.vf v8,  %0, v22" ::"f"(f_[4]));
		asm volatile("vfmacc.vf v6,  %0, v22" ::"f"(f_[7]));

		asm volatile("vslidedown.vi v22, v22, 1");

		asm volatile("vfmacc.vf v12,  %0, v24" ::"f"(f_[1]));
		asm volatile("vfmacc.vf v10,  %0, v24" ::"f"(f_[4]));
		asm volatile("vfmacc.vf v8,  %0, v24" ::"f"(f_[7]));

		asm volatile("vslidedown.vi v24, v24, 1");

		asm volatile("vfmacc.vf v14,  %0, v26" ::"f"(f_[1]));
		asm volatile("vfmacc.vf v12,  %0, v26" ::"f"(f_[4]));
		asm volatile("vfmacc.vf v10,  %0, v26" ::"f"(f_[7]));

		asm volatile("vslidedown.vi v26, v26, 1");

		asm volatile("vfmacc.vf v4,  %0, v16" ::"f"(f_[2]));
		asm volatile("vfmacc.vf v2,  %0, v16" ::"f"(f_[5]));
		asm volatile("vfmacc.vf v0,  %0, v16" ::"f"(f_[8]));


		asm volatile("vfmacc.vf v6,  %0, v18" ::"f"(f_[2]));
		asm volatile("vfmacc.vf v4,  %0, v18" ::"f"(f_[5]));
		asm volatile("vfmacc.vf v2,  %0, v18" ::"f"(f_[8]));


		asm volatile("vfmacc.vf v8,  %0, v20" ::"f"(f_[2]));
		asm volatile("vfmacc.vf v6,  %0, v20" ::"f"(f_[5]));
		asm volatile("vfmacc.vf v4,  %0, v20" ::"f"(f_[8]));


		asm volatile("vfmacc.vf v10,  %0, v22" ::"f"(f_[2]));
		asm volatile("vfmacc.vf v8,  %0, v22" ::"f"(f_[5]));
		asm volatile("vfmacc.vf v6,  %0, v22" ::"f"(f_[8]));


		asm volatile("vfmacc.vf v12,  %0, v24" ::"f"(f_[2]));
		asm volatile("vfmacc.vf v10,  %0, v24" ::"f"(f_[5]));
		asm volatile("vfmacc.vf v8,  %0, v24" ::"f"(f_[8]));


		asm volatile("vfmacc.vf v14,  %0, v26" ::"f"(f_[2]));
		asm volatile("vfmacc.vf v12,  %0, v26" ::"f"(f_[5]));
		asm volatile("vfmacc.vf v10,  %0, v26" ::"f"(f_[8]));



		for(int ch = 1 ; ch < W ; ch ++){

			f_ += 9;

			asm volatile("vle32.v v16, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
			asm volatile("vle32.v v18, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));

			asm volatile("vfmacc.vf v4,  %0, v16" ::"f"(f_[0]));
			asm volatile("vfmacc.vf v2,  %0, v16" ::"f"(f_[3]));
			asm volatile("vfmacc.vf v0,  %0, v16" ::"f"(f_[6]));

			asm volatile("vslidedown.vi v16, v16, 1");

			asm volatile("vle32.v v20, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));

			asm volatile("vfmacc.vf v6,  %0, v18" ::"f"(f_[0]));
			asm volatile("vfmacc.vf v4,  %0, v18" ::"f"(f_[3]));
			asm volatile("vfmacc.vf v2,  %0, v18" ::"f"(f_[6]));

			asm volatile("vslidedown.vi v18, v18, 1");

			asm volatile("vle32.v v22, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));

			asm volatile("vfmacc.vf v8,  %0, v20" ::"f"(f_[0]));
			asm volatile("vfmacc.vf v6,  %0, v20" ::"f"(f_[3]));
			asm volatile("vfmacc.vf v4,  %0, v20" ::"f"(f_[6]));

			asm volatile("vslidedown.vi v20, v20, 1");

			asm volatile("vle32.v v24, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));

			asm volatile("vfmacc.vf v10,  %0, v22" ::"f"(f_[0]));
			asm volatile("vfmacc.vf v8,  %0, v22" ::"f"(f_[3]));
			asm volatile("vfmacc.vf v6,  %0, v22" ::"f"(f_[6]));

			asm volatile("vslidedown.vi v22, v22, 1");

			asm volatile("vle32.v v26, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(next_plane_(6)));

			asm volatile("vfmacc.vf v12,  %0, v24" ::"f"(f_[0]));
			asm volatile("vfmacc.vf v10,  %0, v24" ::"f"(f_[3]));
			asm volatile("vfmacc.vf v8,  %0, v24" ::"f"(f_[6]));

			asm volatile("vslidedown.vi v24, v24, 1");

			asm volatile("vfmacc.vf v14,  %0, v26" ::"f"(f_[0]));
			asm volatile("vfmacc.vf v12,  %0, v26" ::"f"(f_[3]));
			asm volatile("vfmacc.vf v10,  %0, v26" ::"f"(f_[6]));

			asm volatile("vslidedown.vi v26, v26, 1");

			asm volatile("vfmacc.vf v4,  %0, v16" ::"f"(f_[1]));
			asm volatile("vfmacc.vf v2,  %0, v16" ::"f"(f_[4]));
			asm volatile("vfmacc.vf v0,  %0, v16" ::"f"(f_[7]));

			asm volatile("vslidedown.vi v16, v16, 1");

			asm volatile("vfmacc.vf v6,  %0, v18" ::"f"(f_[1]));
			asm volatile("vfmacc.vf v4,  %0, v18" ::"f"(f_[4]));
			asm volatile("vfmacc.vf v2,  %0, v18" ::"f"(f_[7]));

			asm volatile("vslidedown.vi v18, v18, 1");

			asm volatile("vfmacc.vf v8,  %0, v20" ::"f"(f_[1]));
			asm volatile("vfmacc.vf v6,  %0, v20" ::"f"(f_[4]));
			asm volatile("vfmacc.vf v4,  %0, v20" ::"f"(f_[7]));

			asm volatile("vslidedown.vi v20, v20, 1");

			asm volatile("vfmacc.vf v10,  %0, v22" ::"f"(f_[1]));
			asm volatile("vfmacc.vf v8,  %0, v22" ::"f"(f_[4]));
			asm volatile("vfmacc.vf v6,  %0, v22" ::"f"(f_[7]));

			asm volatile("vslidedown.vi v22, v22, 1");

			asm volatile("vfmacc.vf v12,  %0, v24" ::"f"(f_[1]));
			asm volatile("vfmacc.vf v10,  %0, v24" ::"f"(f_[4]));
			asm volatile("vfmacc.vf v8,  %0, v24" ::"f"(f_[7]));

			asm volatile("vslidedown.vi v24, v24, 1");

			asm volatile("vfmacc.vf v14,  %0, v26" ::"f"(f_[1]));
			asm volatile("vfmacc.vf v12,  %0, v26" ::"f"(f_[4]));
			asm volatile("vfmacc.vf v10,  %0, v26" ::"f"(f_[7]));

			asm volatile("vslidedown.vi v26, v26, 1");

			asm volatile("vfmacc.vf v4,  %0, v16" ::"f"(f_[2]));
			asm volatile("vfmacc.vf v2,  %0, v16" ::"f"(f_[5]));
			asm volatile("vfmacc.vf v0,  %0, v16" ::"f"(f_[8]));


			asm volatile("vfmacc.vf v6,  %0, v18" ::"f"(f_[2]));
			asm volatile("vfmacc.vf v4,  %0, v18" ::"f"(f_[5]));
			asm volatile("vfmacc.vf v2,  %0, v18" ::"f"(f_[8]));


			asm volatile("vfmacc.vf v8,  %0, v20" ::"f"(f_[2]));
			asm volatile("vfmacc.vf v6,  %0, v20" ::"f"(f_[5]));
			asm volatile("vfmacc.vf v4,  %0, v20" ::"f"(f_[8]));


			asm volatile("vfmacc.vf v10,  %0, v22" ::"f"(f_[2]));
			asm volatile("vfmacc.vf v8,  %0, v22" ::"f"(f_[5]));
			asm volatile("vfmacc.vf v6,  %0, v22" ::"f"(f_[8]));


			asm volatile("vfmacc.vf v12,  %0, v24" ::"f"(f_[2]));
			asm volatile("vfmacc.vf v10,  %0, v24" ::"f"(f_[5]));
			asm volatile("vfmacc.vf v8,  %0, v24" ::"f"(f_[8]));


			asm volatile("vfmacc.vf v14,  %0, v26" ::"f"(f_[2]));
			asm volatile("vfmacc.vf v12,  %0, v26" ::"f"(f_[5]));
			asm volatile("vfmacc.vf v10,  %0, v26" ::"f"(f_[8]));


			}

		asm volatile("vsetvli zero, %0, e32, m2, ta, ma" ::"r"(vlen_out));


			i__ = i_ + r * C;
			f_ = f;
		  	
		asm volatile("vse32.v v0, (%0); add %0, %0, %1" : "+&r"(o_) : "r"(ldo));
		asm volatile("vse32.v v2, (%0); add %0, %0, %1" : "+&r"(o_) : "r"(ldo));
		asm volatile("vse32.v v4, (%0); add %0, %0, %1" : "+&r"(o_) : "r"(ldo));
		asm volatile("vse32.v v6, (%0); add %0, %0, %1" : "+&r"(o_) : "r"(ldo));
		asm volatile("vse32.v v8, (%0); add %0, %0, %1" : "+&r"(o_) : "r"(ldo));
		asm volatile("vse32.v v10, (%0); add %0, %0, %1" : "+&r"(o_) : "r"(ldo));

		asm volatile("vmv.v.v v0, v12");
		asm volatile("vmv.v.v v2, v14");

		}


	asm volatile("vsetvli zero, %0, e32, m2, ta, ma" ::"r"(vlen));

	if (last_group == 0)
	{
		asm volatile("vle32.v v16, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
		asm volatile("vle32.v v18, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
		asm volatile("vle32.v v20, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
		asm volatile("vle32.v v22, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
		asm volatile("vle32.v v24, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
		asm volatile("vle32.v v26, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(next_plane_(6)));

	}
	else if (last_group == 5)
	{
		asm volatile("vle32.v v16, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
		asm volatile("vle32.v v18, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
		asm volatile("vle32.v v20, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
		asm volatile("vle32.v v22, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
		asm volatile("vle32.v v24, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(next_plane_(5)));

	}
	else if (last_group == 4)
	{
		asm volatile("vle32.v v16, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
		asm volatile("vle32.v v18, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
		asm volatile("vle32.v v20, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
		asm volatile("vle32.v v22, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(next_plane_(4)));

	}
	else if (last_group == 3)
	{
		asm volatile("vle32.v v16, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
		asm volatile("vle32.v v18, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
		asm volatile("vle32.v v20, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(next_plane_(3)));

	}
	else if (last_group == 2)
	{
		asm volatile("vle32.v v16, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
		asm volatile("vle32.v v18, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(next_plane_(2)));

	}
	else
	{
		asm volatile("vle32.v v16, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(next_plane_(1)));

	}
	asm volatile("vfmacc.vf v0,  %0, v16" ::"f"(f_[6]));
	asm volatile("vfmacc.vf v2,  %0, v18" ::"f"(f_[6]));
	asm volatile("vfmul.vf v4,  v20, %0" :: "f"(f_[6]));
	asm volatile("vfmul.vf v6,  v22, %0" :: "f"(f_[6]));
	asm volatile("vfmul.vf v8,  v24, %0" :: "f"(f_[6]));
	asm volatile("vfmul.vf v10,  v26, %0" :: "f"(f_[6]));

	asm volatile("vslidedown.vi v26, v26, 1");
	asm volatile("vfmacc.vf v2,  %0, v16" ::"f"(f_[3]));
	asm volatile("vfmacc.vf v4,  %0, v18" ::"f"(f_[3]));
	asm volatile("vfmacc.vf v6,  %0, v20" ::"f"(f_[3]));
	asm volatile("vfmacc.vf v8,  %0, v22" ::"f"(f_[3]));
	asm volatile("vfmacc.vf v10,  %0, v24" ::"f"(f_[3]));

	asm volatile("vslidedown.vi v24, v24, 1");
	asm volatile("vfmacc.vf v4,  %0, v16" ::"f"(f_[0]));
	asm volatile("vfmacc.vf v6,  %0, v18" ::"f"(f_[0]));
	asm volatile("vfmacc.vf v8,  %0, v20" ::"f"(f_[0]));
	asm volatile("vfmacc.vf v10,  %0, v22" ::"f"(f_[0]));

	asm volatile("vslidedown.vi v22, v22, 1");
	asm volatile("vslidedown.vi v20, v20, 1");
	asm volatile("vslidedown.vi v18, v18, 1");
	asm volatile("vslidedown.vi v16, v16, 1");

	asm volatile("vfmacc.vf v0,  %0, v16" ::"f"(f_[7]));
	asm volatile("vfmacc.vf v2,  %0, v18" ::"f"(f_[7]));
	asm volatile("vfmacc.vf v4,  %0, v20" ::"f"(f_[7]));
	asm volatile("vfmacc.vf v6,  %0, v22" ::"f"(f_[7]));
	asm volatile("vfmacc.vf v8,  %0, v24" ::"f"(f_[7]));
	asm volatile("vfmacc.vf v10,  %0, v26" ::"f"(f_[7]));

	asm volatile("vslidedown.vi v26, v26, 1");
	asm volatile("vfmacc.vf v2,  %0, v16" ::"f"(f_[4]));
	asm volatile("vfmacc.vf v4,  %0, v18" ::"f"(f_[4]));
	asm volatile("vfmacc.vf v6,  %0, v20" ::"f"(f_[4]));
	asm volatile("vfmacc.vf v8,  %0, v22" ::"f"(f_[4]));
	asm volatile("vfmacc.vf v10,  %0, v24" ::"f"(f_[4]));

	asm volatile("vslidedown.vi v24, v24, 1");
	asm volatile("vfmacc.vf v4,  %0, v16" ::"f"(f_[1]));
	asm volatile("vfmacc.vf v6,  %0, v18" ::"f"(f_[1]));
	asm volatile("vfmacc.vf v8,  %0, v20" ::"f"(f_[1]));
	asm volatile("vfmacc.vf v10,  %0, v22" ::"f"(f_[1]));

	asm volatile("vslidedown.vi v22, v22, 1");
	asm volatile("vslidedown.vi v20, v20, 1");
	asm volatile("vslidedown.vi v18, v18, 1");
	asm volatile("vslidedown.vi v16, v16, 1");

	asm volatile("vfmacc.vf v0,  %0, v16" ::"f"(f_[8]));
	asm volatile("vfmacc.vf v2,  %0, v18" ::"f"(f_[8]));
	asm volatile("vfmacc.vf v4,  %0, v20" ::"f"(f_[8]));
	asm volatile("vfmacc.vf v6,  %0, v22" ::"f"(f_[8]));
	asm volatile("vfmacc.vf v8,  %0, v24" ::"f"(f_[8]));
	asm volatile("vfmacc.vf v10,  %0, v26" ::"f"(f_[8]));

	asm volatile("vfmacc.vf v2,  %0, v16" ::"f"(f_[5]));
	asm volatile("vfmacc.vf v4,  %0, v18" ::"f"(f_[5]));
	asm volatile("vfmacc.vf v6,  %0, v20" ::"f"(f_[5]));
	asm volatile("vfmacc.vf v8,  %0, v22" ::"f"(f_[5]));
	asm volatile("vfmacc.vf v10,  %0, v24" ::"f"(f_[5]));

	asm volatile("vfmacc.vf v4,  %0, v16" ::"f"(f_[2]));
	asm volatile("vfmacc.vf v6,  %0, v18" ::"f"(f_[2]));
	asm volatile("vfmacc.vf v8,  %0, v20" ::"f"(f_[2]));
	asm volatile("vfmacc.vf v10,  %0, v22" ::"f"(f_[2]));



	for(int ch = 1 ; ch < W ; ch ++){

		f_ += 9;

		if (last_group == 0)
		{
			asm volatile("vle32.v v16, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
			asm volatile("vle32.v v18, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
			asm volatile("vle32.v v20, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
			asm volatile("vle32.v v22, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
			asm volatile("vle32.v v24, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
			asm volatile("vle32.v v26, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(next_plane_(6)));

		}
		else if (last_group == 5)
		{
			asm volatile("vle32.v v16, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
			asm volatile("vle32.v v18, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
			asm volatile("vle32.v v20, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
			asm volatile("vle32.v v22, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
			asm volatile("vle32.v v24, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(next_plane_(5)));

		}
		else if (last_group == 4)
		{
			asm volatile("vle32.v v16, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
			asm volatile("vle32.v v18, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
			asm volatile("vle32.v v20, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
			asm volatile("vle32.v v22, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(next_plane_(4)));

		}
		else if (last_group == 3)
		{
			asm volatile("vle32.v v16, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
			asm volatile("vle32.v v18, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
			asm volatile("vle32.v v20, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(next_plane_(3)));

		}
		else if (last_group == 2)
		{
			asm volatile("vle32.v v16, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
			asm volatile("vle32.v v18, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(next_plane_(2)));

		}
		else
		{
			asm volatile("vle32.v v16, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(next_plane_(1)));

		}
		asm volatile("vfmacc.vf v0,  %0, v16" ::"f"(f_[6]));
		asm volatile("vfmacc.vf v2,  %0, v18" ::"f"(f_[6]));
		asm volatile("vfmacc.vf v4,  %0, v20" ::"f"(f_[6]));
		asm volatile("vfmacc.vf v6,  %0, v22" ::"f"(f_[6]));
		asm volatile("vfmacc.vf v8,  %0, v24" ::"f"(f_[6]));
		asm volatile("vfmacc.vf v10,  %0, v26" ::"f"(f_[6]));

		asm volatile("vslidedown.vi v26, v26, 1");
		asm volatile("vfmacc.vf v2,  %0, v16" ::"f"(f_[3]));
		asm volatile("vfmacc.vf v4,  %0, v18" ::"f"(f_[3]));
		asm volatile("vfmacc.vf v6,  %0, v20" ::"f"(f_[3]));
		asm volatile("vfmacc.vf v8,  %0, v22" ::"f"(f_[3]));
		asm volatile("vfmacc.vf v10,  %0, v24" ::"f"(f_[3]));

		asm volatile("vslidedown.vi v24, v24, 1");
		asm volatile("vfmacc.vf v4,  %0, v16" ::"f"(f_[0]));
		asm volatile("vfmacc.vf v6,  %0, v18" ::"f"(f_[0]));
		asm volatile("vfmacc.vf v8,  %0, v20" ::"f"(f_[0]));
		asm volatile("vfmacc.vf v10,  %0, v22" ::"f"(f_[0]));

		asm volatile("vslidedown.vi v22, v22, 1");
		asm volatile("vslidedown.vi v20, v20, 1");
		asm volatile("vslidedown.vi v18, v18, 1");
		asm volatile("vslidedown.vi v16, v16, 1");

		asm volatile("vfmacc.vf v0,  %0, v16" ::"f"(f_[7]));
		asm volatile("vfmacc.vf v2,  %0, v18" ::"f"(f_[7]));
		asm volatile("vfmacc.vf v4,  %0, v20" ::"f"(f_[7]));
		asm volatile("vfmacc.vf v6,  %0, v22" ::"f"(f_[7]));
		asm volatile("vfmacc.vf v8,  %0, v24" ::"f"(f_[7]));
		asm volatile("vfmacc.vf v10,  %0, v26" ::"f"(f_[7]));

		asm volatile("vslidedown.vi v26, v26, 1");
		asm volatile("vfmacc.vf v2,  %0, v16" ::"f"(f_[4]));
		asm volatile("vfmacc.vf v4,  %0, v18" ::"f"(f_[4]));
		asm volatile("vfmacc.vf v6,  %0, v20" ::"f"(f_[4]));
		asm volatile("vfmacc.vf v8,  %0, v22" ::"f"(f_[4]));
		asm volatile("vfmacc.vf v10,  %0, v24" ::"f"(f_[4]));

		asm volatile("vslidedown.vi v24, v24, 1");
		asm volatile("vfmacc.vf v4,  %0, v16" ::"f"(f_[1]));
		asm volatile("vfmacc.vf v6,  %0, v18" ::"f"(f_[1]));
		asm volatile("vfmacc.vf v8,  %0, v20" ::"f"(f_[1]));
		asm volatile("vfmacc.vf v10,  %0, v22" ::"f"(f_[1]));

		asm volatile("vslidedown.vi v22, v22, 1");
		asm volatile("vslidedown.vi v20, v20, 1");
		asm volatile("vslidedown.vi v18, v18, 1");
		asm volatile("vslidedown.vi v16, v16, 1");

		asm volatile("vfmacc.vf v0,  %0, v16" ::"f"(f_[8]));
		asm volatile("vfmacc.vf v2,  %0, v18" ::"f"(f_[8]));
		asm volatile("vfmacc.vf v4,  %0, v20" ::"f"(f_[8]));
		asm volatile("vfmacc.vf v6,  %0, v22" ::"f"(f_[8]));
		asm volatile("vfmacc.vf v8,  %0, v24" ::"f"(f_[8]));
		asm volatile("vfmacc.vf v10,  %0, v26" ::"f"(f_[8]));

		asm volatile("vfmacc.vf v2,  %0, v16" ::"f"(f_[5]));
		asm volatile("vfmacc.vf v4,  %0, v18" ::"f"(f_[5]));
		asm volatile("vfmacc.vf v6,  %0, v20" ::"f"(f_[5]));
		asm volatile("vfmacc.vf v8,  %0, v22" ::"f"(f_[5]));
		asm volatile("vfmacc.vf v10,  %0, v24" ::"f"(f_[5]));

		asm volatile("vfmacc.vf v4,  %0, v16" ::"f"(f_[2]));
		asm volatile("vfmacc.vf v6,  %0, v18" ::"f"(f_[2]));
		asm volatile("vfmacc.vf v8,  %0, v20" ::"f"(f_[2]));
		asm volatile("vfmacc.vf v10,  %0, v22" ::"f"(f_[2]));


		}

	asm volatile("vsetvli zero, %0, e32, m2, ta, ma" ::"r"(vlen_out));

	if (last_group == 0)
	{
	asm volatile("vse32.v v0, (%0); add %0, %0, %1" : "+&r"(o_) : "r"(ldo));
	asm volatile("vse32.v v2, (%0); add %0, %0, %1" : "+&r"(o_) : "r"(ldo));
	asm volatile("vse32.v v4, (%0); add %0, %0, %1" : "+&r"(o_) : "r"(ldo));
	asm volatile("vse32.v v6, (%0); add %0, %0, %1" : "+&r"(o_) : "r"(ldo));
	asm volatile("vse32.v v8, (%0); add %0, %0, %1" : "+&r"(o_) : "r"(ldo));
	asm volatile("vse32.v v10, (%0)" : "+&r"(o_));

	}
	else if (last_group == 5)
	{
	asm volatile("vse32.v v0, (%0); add %0, %0, %1" : "+&r"(o_) : "r"(ldo));
	asm volatile("vse32.v v2, (%0); add %0, %0, %1" : "+&r"(o_) : "r"(ldo));
	asm volatile("vse32.v v4, (%0); add %0, %0, %1" : "+&r"(o_) : "r"(ldo));
	asm volatile("vse32.v v6, (%0); add %0, %0, %1" : "+&r"(o_) : "r"(ldo));
	asm volatile("vse32.v v8, (%0)" : "+&r"(o_));

	}
	else if (last_group == 4)
	{
	asm volatile("vse32.v v0, (%0); add %0, %0, %1" : "+&r"(o_) : "r"(ldo));
	asm volatile("vse32.v v2, (%0); add %0, %0, %1" : "+&r"(o_) : "r"(ldo));
	asm volatile("vse32.v v4, (%0); add %0, %0, %1" : "+&r"(o_) : "r"(ldo));
	asm volatile("vse32.v v6, (%0)" : "+&r"(o_));

	}
	else if (last_group == 3)
	{
	asm volatile("vse32.v v0, (%0); add %0, %0, %1" : "+&r"(o_) : "r"(ldo));
	asm volatile("vse32.v v2, (%0); add %0, %0, %1" : "+&r"(o_) : "r"(ldo));
	asm volatile("vse32.v v4, (%0)" : "+&r"(o_));

	}
	else if (last_group == 2)
	{
	asm volatile("vse32.v v0, (%0); add %0, %0, %1" : "+&r"(o_) : "r"(ldo));
	asm volatile("vse32.v v2, (%0)" : "+&r"(o_));

	}
	else
	{
	asm volatile("vse32.v v0, (%0)" : "+&r"(o_));

	}
	}
}