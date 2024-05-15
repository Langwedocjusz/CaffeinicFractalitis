#include "ComputeFractalSSE.h"

#include <cmath>
#include <array>

void ComputeFractalSSE::Mandelbrot(float* mem_address, __m128  x, __m128 y)
{
    constexpr size_t iter_max = 400;
    constexpr float bailout = 100.0f;
	
    __m128 re, im, r2, i2, len2;
	__m128 condition, iter;
	__m128 final_len2;

	re = _mm_setzero_ps();
	im = _mm_setzero_ps();
	r2 = _mm_setzero_ps();
	i2 = _mm_setzero_ps();

	condition = _mm_setzero_ps();
	iter = _mm_setzero_ps();
	//Since smooth iteration count is computed using the modulus of last position,
	//we need additional variable for this, as re and im will all be iterated
	//until all four bailout conditions are met
	final_len2 = _mm_setzero_ps(); 

	const __m128 one = _mm_set1_ps(1.0f);
	const __m128 two = _mm_set1_ps(2.0f);

	const __m128 bail2 = _mm_set1_ps(bailout*bailout);

	for (size_t k = 0; k < iter_max; k++)
	{
		im = _mm_mul_ps(re, im);
		im = _mm_mul_ps(im, two);
		im = _mm_add_ps(im, y);

		re = _mm_add_ps(_mm_sub_ps(r2, i2), x);

		r2 = _mm_mul_ps(re, re);
		i2 = _mm_mul_ps(im, im);

		len2 = _mm_add_ps(r2, i2);
		//Copy len2's of only those pixels that are yet to bail out
		final_len2 = _mm_blendv_ps(len2, final_len2, condition);
			
		condition = _mm_or_ps(condition, _mm_cmpgt_ps(len2, bail2));

		iter = _mm_add_ps(iter, _mm_andnot_ps(condition, one));

		if (_mm_movemask_ps(condition) == 0x0f)
			break;
	}
	
	//Save result
	_mm_store_ps(mem_address, iter);

	//Smooth interation count
	std::array<float, 4> moduli;
	_mm_store_ps(&moduli[0], final_len2);
	
	constexpr float deg = 2.0f;
	const float inv_log_bail = 1.0f / std::log(bailout);
	const float sm_inv_denom = 1.0f / std::log(deg);

	for (size_t i = 0; i < 4; i++)
	{
		const float smoothing = sm_inv_denom 
            * std::log(0.5f * inv_log_bail * std::log(moduli[i]));

		*(mem_address + i) -= smoothing;
	}
}


void ComputeFractalSSE::MandelbrotLight(float* mem_address, __m128 x, __m128 y)
{
    constexpr size_t iter_max = 400;
    constexpr float bailout = 100.0f;
    constexpr float light_height = 1.5f;
    constexpr float l_x = 0.7071f, l_y = 0.7071f;

    __m128 re, im, r2, i2, len2;
	__m128 d_re, d_im;
    __m128 condition;

	__m128 final_re, final_im;
    __m128 final_d_re, final_d_im;

    re = _mm_setzero_ps();
    im = _mm_setzero_ps();
    r2 = _mm_setzero_ps();
    i2 = _mm_setzero_ps();
    d_re = _mm_setzero_ps();
    d_im = _mm_setzero_ps();
    condition = _mm_setzero_ps();
    final_re = _mm_setzero_ps();
    final_im = _mm_setzero_ps();
    final_d_re = _mm_setzero_ps();
    final_d_im = _mm_setzero_ps();

    const __m128 zero = _mm_set1_ps(0.0f);
	const __m128 one = _mm_set1_ps(1.0f);
	const __m128 two = _mm_set1_ps(2.0f);

	const __m128 bail2 = _mm_set1_ps(bailout*bailout);
	const __m128 lx = _mm_set1_ps(l_x);
	const __m128 ly = _mm_set1_ps(l_y);
    const __m128 lheight = _mm_set1_ps(light_height);

	for (size_t k = 0; k < iter_max; k++)
	{
        //z -> z^2
		__m128 new_im = _mm_mul_ps(re, im);
		new_im = _mm_mul_ps(new_im, two);
		new_im = _mm_add_ps(new_im, y);

		__m128 new_re = _mm_add_ps(_mm_sub_ps(r2, i2), x);

        //dz -> 2zdz + 1
        __m128 new_d_re = _mm_add_ps(
			_mm_mul_ps(
                _mm_sub_ps(_mm_mul_ps(re, d_re), _mm_mul_ps(im, d_im)), 
        	two), 
		one);

        __m128 new_d_im = _mm_mul_ps(_mm_add_ps(_mm_mul_ps(re, d_im), _mm_mul_ps(im, d_re)), two);

        re = new_re;
        im = new_im;
        d_re = new_d_re;
        d_im = new_d_im;

		r2 = _mm_mul_ps(re, re);
		i2 = _mm_mul_ps(im, im);

		len2 = _mm_add_ps(r2, i2);

		//Save values of only those pixels that are yet to bail out
		final_re = _mm_blendv_ps(re, final_re, condition);
		final_im = _mm_blendv_ps(im, final_im, condition);
		final_d_re = _mm_blendv_ps(d_re, final_d_re, condition);
		final_d_im = _mm_blendv_ps(d_im, final_d_im, condition);
			
		condition = _mm_or_ps(condition, _mm_cmpgt_ps(len2, bail2));

		if (_mm_movemask_ps(condition) == 0x0f)
			break;
	}

    const __m128 dlen2 = _mm_add_ps(
        _mm_mul_ps(final_d_re, final_d_re), 
        _mm_mul_ps(final_d_im, final_d_im)
    );

    //u = z/dz (complex division)
    __m128 u_re = _mm_div_ps(_mm_add_ps(_mm_mul_ps(final_re, final_d_re), _mm_mul_ps(final_im, final_d_im)), dlen2);
    __m128 u_im = _mm_div_ps(_mm_sub_ps(_mm_mul_ps(final_im, final_d_re), _mm_mul_ps(final_re, final_d_im)), dlen2);

    const __m128 u_len = _mm_sqrt_ps(_mm_add_ps(_mm_mul_ps(u_re, u_re), _mm_mul_ps(u_im, u_im)));

    u_re = _mm_div_ps(u_re, u_len);
    u_im = _mm_div_ps(u_im, u_len);

    const __m128 dot = _mm_add_ps(_mm_add_ps(_mm_mul_ps(u_re, lx), _mm_mul_ps(u_im, ly)), lheight);

    __m128 res = _mm_max_ps(zero, _mm_min_ps(_mm_div_ps(dot, _mm_add_ps(one, lheight)), one));

    res = _mm_blendv_ps(zero, res, condition);

	_mm_store_ps(mem_address, res);
}
