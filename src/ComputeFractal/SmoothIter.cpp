#include "SmoothIter.h"

#include <cmath>
#include <array>

#include "ComplexArithmetic.h"

float ComputeFractal::SmoothIter(float x, float y)
{
	using enum SimdType;
	using complex = Complex<Scalar>;

    constexpr size_t iter_max = 400;
    constexpr float bailout = 8.0f;

	const complex c(x, y);

	complex z(0.0f, 0.0f);

	float len2 = 0.0f;
	float iterations = 0.0f;

	for (size_t k = 0; k < iter_max; k++)
	{
		z = z*z + c;

		len2 = complex::Len2(z).Value;

		if (len2 > bailout*bailout) break;

		iterations += 1.0f;
	}

	constexpr float deg = 2.0f;
	const float inv_log_bail = 1.0f / std::log(bailout);
	const float sm_inv_denom = 1.0f / std::log(deg);

	const float smoothing = sm_inv_denom 
        * std::log(0.5f * std::log(len2) * inv_log_bail);

	return iterations - smoothing;
}

void ComputeFractal::SmoothIterSSE(float* mem_address, __m128  x, __m128 y)
{
	using enum SimdType;
	using complex = Complex<SSE>;

    constexpr size_t iter_max = 400;
    constexpr float bailout = 100.0f;

	const complex c(x, y);

	complex z(0.0f, 0.0f);

	__m128 condition = _mm_setzero_ps();
	__m128 iter = _mm_setzero_ps();

	//Since smooth iteration count is computed using the modulus of last position,
	//we need additional variable for this, as re and im will all be iterated
	//until all four bailout conditions are met
	__m128 final_len2 = _mm_setzero_ps(); 

	const __m128 one = _mm_set1_ps(1.0f);
	const __m128 bail2 = _mm_set1_ps(bailout*bailout);

	for (size_t k = 0; k < iter_max; k++)
	{
		z = z*z + c;

		const __m128 len2 = complex::Len2(z).Value;

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

void ComputeFractal::SmoothIterAVX(float* mem_address, __m256  x, __m256 y)
{
	using enum SimdType;
	using complex = Complex<AVX>;

    constexpr size_t iter_max = 400;
    constexpr float bailout = 100.0f;

	const complex c(x, y);

	complex z(0.0f, 0.0f);

	__m256 condition = _mm256_setzero_ps();
	__m256 iter = _mm256_setzero_ps();

	//Since smooth iteration count is computed using the modulus of last position,
	//we need additional variable for this, as re and im will all be iterated
	//until all four bailout conditions are met
	__m256 final_len2 = _mm256_setzero_ps(); 

	const __m256 one = _mm256_set1_ps(1.0f);
	const __m256 bail2 = _mm256_set1_ps(bailout*bailout);

	for (size_t k = 0; k < iter_max; k++)
	{
		z = z*z + c;

		const __m256 len2 = complex::Len2(z).Value;

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