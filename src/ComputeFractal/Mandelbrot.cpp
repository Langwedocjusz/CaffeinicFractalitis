#include "Mandelbrot.h"

#include <cmath>
#include <array>

float ComputeFractal::Mandelbrot(float x, float y)
{
    constexpr size_t iter_max = 400;
    constexpr float bailout = 8.0f;

	float re = 0.0f, im = 0.0f, r2 = 0.0f, i2 = 0.0f;

	float iterations = 0.0f;

	for (size_t k = 0; k < iter_max; k++)
	{
		im = 2.0f * re * im + y;
		re = r2 - i2        + x;
		
		r2 = re * re;
		i2 = im * im;

		if ((r2 + i2) > bailout*bailout) break;

		iterations += 1.0f;
	}

	const float len2 = r2 + i2;

	constexpr float deg = 2.0f;
	const float inv_log_bail = 1.0f / std::log(bailout);
	const float sm_inv_denom = 1.0f / std::log(deg);

	const float smoothing = sm_inv_denom 
        * std::log(0.5f * std::log(len2) * inv_log_bail);

	return iterations - smoothing;
}

void ComputeFractal::MandelbrotSSE(float* mem_address, __m128  x, __m128 y)
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

void ComputeFractal::MandelbrotAVX(float* mem_address, __m256  x, __m256 y)
{
    constexpr size_t iter_max = 400;
    constexpr float bailout = 100.0f;
	
    __m256 re, im, r2, i2, len2;
	__m256 condition, iter;
	__m256 final_len2;

	re = _mm256_setzero_ps();
	im = _mm256_setzero_ps();
	r2 = _mm256_setzero_ps();
	i2 = _mm256_setzero_ps();

	condition = _mm256_setzero_ps();
	iter = _mm256_setzero_ps();
	//Since smooth iteration count is computed using the modulus of last position,
	//we need additional variable for this, as re and im will all be iterated
	//until all four bailout conditions are met
	final_len2 = _mm256_setzero_ps(); 

	const __m256 one = _mm256_set1_ps(1.0f);
	const __m256 two = _mm256_set1_ps(2.0f);

	const __m256 bail2 = _mm256_set1_ps(bailout*bailout);

	for (size_t k = 0; k < iter_max; k++)
	{
		im = _mm256_mul_ps(re, im);
		im = _mm256_mul_ps(im, two);
		im = _mm256_add_ps(im, y);

		re = _mm256_add_ps(_mm256_sub_ps(r2, i2), x);

		r2 = _mm256_mul_ps(re, re);
		i2 = _mm256_mul_ps(im, im);

		len2 = _mm256_add_ps(r2, i2);
		//Copy len2's of only those pixels that are yet to bail out
		final_len2 = _mm256_blendv_ps(len2, final_len2, condition);
			
		condition = _mm256_or_ps(condition, _mm256_cmp_ps(len2, bail2, _CMP_GT_OS));

		iter = _mm256_add_ps(iter, _mm256_andnot_ps(condition, one));

		if (_mm256_movemask_ps(condition) == 0x0f)
			break;
	}
	
	//Save result
	_mm256_store_ps(mem_address, iter);

	//Smooth interation count
	std::array<float, 8> moduli;
	_mm256_store_ps(&moduli[0], final_len2);
	
	constexpr float deg = 2.0f;
	const float inv_log_bail = 1.0f / std::log(bailout);
	const float sm_inv_denom = 1.0f / std::log(deg);

	for (size_t i = 0; i < 8; i++)
	{
		const float smoothing = sm_inv_denom 
            * std::log(0.5f * inv_log_bail * std::log(moduli[i]));
	
		*(mem_address + i) -= smoothing;
	}
}