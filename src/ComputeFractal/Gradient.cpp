#include "Gradient.h"

#include <cmath>

#include "ComplexArithmetic.h"

float ComputeFractal::Gradient(float x, float y)
{
    using enum SimdType;
	using complex = Complex<Scalar>;

    constexpr size_t iter_max = 400;
    constexpr float bailout = 100.0f;
    constexpr float light_height = 1.5f;

    const complex l(0.7071f, 0.7071f);
    const complex c(x, y);

    complex z(0.0f, 0.0f);
    complex dz(0.0f, 0.0f);

    bool not_enough_iterations = true;

	for (size_t k = 0; k < iter_max; k++)
	{
        const complex new_z = z*z + c;
        dz = 2.0f * z * dz + 1.0f;
        z = new_z;

        const float len2 = complex::Len2(z).Value;

		if (len2 > bailout*bailout)
        {
            not_enough_iterations = false;
            break;
        }
	}

    if (not_enough_iterations)
        return -1.0f;

    complex u = z/dz;
    u /= complex::Len(u);

    const float dot = (complex::Dot(u, l) + light_height).Value;
    
    return std::max(0.0f, std::min(dot/(1.0f + light_height), 1.0f));
}

void ComputeFractal::GradientSSE(float* mem_address, __m128 x, __m128 y)
{
    using enum SimdType;
	using complex = Complex<SSE>;

    constexpr size_t iter_max = 400;
    constexpr float bailout = 100.0f;
    constexpr float light_height = 1.5f;

    const complex l(0.7071f, 0.7071f);
    const complex c(x, y);

    complex z(0.0f, 0.0f);
    complex dz(0.0f, 0.0f);

    complex final_z(0.0f, 0.0f);
    complex final_dz(0.0f, 0.0f);

    __m128 condition = _mm_setzero_ps();

    const __m128 zero = _mm_set1_ps(0.0f);
	const __m128 one = _mm_set1_ps(1.0f);

	const __m128 bail2 = _mm_set1_ps(bailout*bailout);
    const __m128 lheight = _mm_set1_ps(light_height);

	for (size_t k = 0; k < iter_max; k++)
	{
        const complex new_z = z*z + c;
        dz = 2.0f * z * dz + 1.0f;
        z = new_z;

        const __m128 len2 = complex::Len2(z).Value;

		//Save values of only those pixels that are yet to bail out
        final_z = complex::Blend(z, final_z, condition);
        final_dz = complex::Blend(dz, final_dz, condition);
			
		condition = _mm_or_ps(condition, _mm_cmpgt_ps(len2, bail2));

		if (_mm_movemask_ps(condition) == 0x0f)
			break;
	}

    complex u = final_z/final_dz;
    u /= complex::Len(u);

    const __m128 dot = (complex::Dot(u, l) + light_height).Value;

    __m128 res = _mm_max_ps(zero, _mm_min_ps(_mm_div_ps(dot, _mm_add_ps(one, lheight)), one));

    res = _mm_blendv_ps(zero, res, condition);

	_mm_store_ps(mem_address, res);
}

void ComputeFractal::GradientAVX(float* mem_address, __m256 x, __m256 y)
{
    using enum SimdType;
	using complex = Complex<AVX>;

    constexpr size_t iter_max = 400;
    constexpr float bailout = 100.0f;
    constexpr float light_height = 1.5f;

    const complex l(0.7071f, 0.7071f);
    const complex c(x, y);

    complex z(0.0f, 0.0f);
    complex dz(0.0f, 0.0f);

    complex final_z(0.0f, 0.0f);
    complex final_dz(0.0f, 0.0f);

    __m256 condition = _mm256_setzero_ps();

    const __m256 zero = _mm256_set1_ps(0.0f);
	const __m256 one = _mm256_set1_ps(1.0f);

	const __m256 bail2 = _mm256_set1_ps(bailout*bailout);
    const __m256 lheight = _mm256_set1_ps(light_height);

	for (size_t k = 0; k < iter_max; k++)
	{
        const complex new_z = z*z + c;
        dz = 2.0f * z * dz + 1.0f;
        z = new_z;

        const __m256 len2 = complex::Len2(z).Value;

		//Save values of only those pixels that are yet to bail out
        final_z = complex::Blend(z, final_z, condition);
        final_dz = complex::Blend(dz, final_dz, condition);
			
		condition = _mm256_or_ps(condition, _mm256_cmp_ps(len2, bail2, _CMP_GT_OS));

		if (_mm256_movemask_ps(condition) == 0x0f)
			break;
	}

    complex u = final_z/final_dz;
    u /= complex::Len(u);

    const __m256 dot = (complex::Dot(u, l) + light_height).Value;

    __m256 res = _mm256_max_ps(zero, _mm256_min_ps(_mm256_div_ps(dot, _mm256_add_ps(one, lheight)), one));

    res = _mm256_blendv_ps(zero, res, condition);

	_mm256_store_ps(mem_address, res);
}