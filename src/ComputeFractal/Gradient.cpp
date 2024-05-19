#include "Gradient.h"

#include <cmath>

float ComputeFractal::Gradient(float x, float y)
{
    constexpr size_t iter_max = 400;
    constexpr float bailout = 100.0f;
    constexpr float light_height = 1.5f;
    constexpr float l_x = 0.7071f, l_y = 0.7071f;

	float re = 0.0f, im = 0.0f, r2 = 0.0f, i2 = 0.0f;
    float d_re = 0.0f, d_im = 0.0f;

    bool not_enough_iterations = true;

	for (size_t k = 0; k < iter_max; k++)
	{
        //z -> z^2
		const float new_im = 2.0f * re * im + y;
		const float new_re = r2 - i2        + x;
		
        //dz -> 2zdz + 1
        const float new_d_re = 2.0f * (re * d_re - im * d_im) + 1.0f;
        const float new_d_im = 2.0f * (re * d_im + im * d_re);

        re = new_re;
        im = new_im;
        d_re = new_d_re;
        d_im = new_d_im;

		r2 = re * re;
		i2 = im * im;

		if ((r2 + i2) > bailout*bailout)
        {
            not_enough_iterations = false;
            break;
        }
	}

    if (not_enough_iterations)
        return -1.0f;

    const float dlen2 = d_re*d_re + d_im*d_im;

    //u = z/dz (complex division)
    float u_re = (re*d_re + im*d_im)/dlen2;
    float u_im = (im * d_re - re* d_im)/dlen2;
    float u_len = std::sqrt(u_re * u_re + u_im * u_im);

    u_re /= u_len;
    u_im /= u_len;

    const float dot = u_re * l_x + u_im * l_y + light_height;
    return std::max(0.0f, std::min(dot/(1.0f + light_height), 1.0f));
}

void ComputeFractal::GradientSSE(float* mem_address, __m128 x, __m128 y)
{
    constexpr size_t iter_max = 400;
    constexpr float bailout = 100.0f;
    constexpr float light_height = 1.5f;
    constexpr float l_x = 0.7071f, l_y = 0.7071f;

    __m128 len2;

    __m128 re = _mm_setzero_ps();
    __m128 im = _mm_setzero_ps();
    __m128 r2 = _mm_setzero_ps();
    __m128 i2 = _mm_setzero_ps();
    __m128 d_re = _mm_setzero_ps();
    __m128 d_im = _mm_setzero_ps();
    __m128 condition = _mm_setzero_ps();
    __m128 final_re = _mm_setzero_ps();
    __m128 final_im = _mm_setzero_ps();
    __m128 final_d_re = _mm_setzero_ps();
    __m128 final_d_im = _mm_setzero_ps();

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

        __m128 new_d_im = _mm_mul_ps(
            _mm_add_ps(_mm_mul_ps(re, d_im), _mm_mul_ps(im, d_re)),
        two);

        re = new_re;
        im = new_im;
        d_re = new_d_re;
        d_im = new_d_im;

		r2 = _mm_mul_ps(re, re);
		i2 = _mm_mul_ps(im, im);

		const __m128 len2 = _mm_add_ps(r2, i2);

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

void ComputeFractal::GradientAVX(float* mem_address, __m256 x, __m256 y)
{
    constexpr size_t iter_max = 400;
    constexpr float bailout = 100.0f;
    constexpr float light_height = 1.5f;
    constexpr float l_x = 0.7071f, l_y = 0.7071f;

    __m256 re = _mm256_setzero_ps();
    __m256 im = _mm256_setzero_ps();
    __m256 r2 = _mm256_setzero_ps();
    __m256 i2 = _mm256_setzero_ps();
    __m256 d_re = _mm256_setzero_ps();
    __m256 d_im = _mm256_setzero_ps();
    __m256 condition = _mm256_setzero_ps();
    __m256 final_re = _mm256_setzero_ps();
    __m256 final_im = _mm256_setzero_ps();
    __m256 final_d_re = _mm256_setzero_ps();
    __m256 final_d_im = _mm256_setzero_ps();

    const __m256 zero = _mm256_set1_ps(0.0f);
	const __m256 one = _mm256_set1_ps(1.0f);
	const __m256 two = _mm256_set1_ps(2.0f);

	const __m256 bail2 = _mm256_set1_ps(bailout*bailout);
	const __m256 lx = _mm256_set1_ps(l_x);
	const __m256 ly = _mm256_set1_ps(l_y);
    const __m256 lheight = _mm256_set1_ps(light_height);

	for (size_t k = 0; k < iter_max; k++)
	{
        //z -> z^2
		__m256 new_im = _mm256_mul_ps(re, im);
		new_im = _mm256_mul_ps(new_im, two);
		new_im = _mm256_add_ps(new_im, y);

		__m256 new_re = _mm256_add_ps(_mm256_sub_ps(r2, i2), x);

        //dz -> 2zdz + 1
        __m256 new_d_re = _mm256_add_ps(_mm256_mul_ps(
                    _mm256_sub_ps(_mm256_mul_ps(re, d_re), _mm256_mul_ps(im, d_im)), 
                    two), one);

        __m256 new_d_im = _mm256_mul_ps(_mm256_add_ps(_mm256_mul_ps(re, d_im), _mm256_mul_ps(im, d_re)), two);

        re = new_re;
        im = new_im;
        d_re = new_d_re;
        d_im = new_d_im;

		r2 = _mm256_mul_ps(re, re);
		i2 = _mm256_mul_ps(im, im);

		const __m256 len2 = _mm256_add_ps(r2, i2);

		//Save values of only those pixels that are yet to bail out
		final_re = _mm256_blendv_ps(re, final_re, condition);
		final_im = _mm256_blendv_ps(im, final_im, condition);
		final_d_re = _mm256_blendv_ps(d_re, final_d_re, condition);
		final_d_im = _mm256_blendv_ps(d_im, final_d_im, condition);
			
		condition = _mm256_or_ps(condition, _mm256_cmp_ps(len2, bail2, _CMP_GT_OS));

		if (_mm256_movemask_ps(condition) == 0x0f)
			break;
	}

    const __m256 dlen2 = _mm256_add_ps(
        _mm256_mul_ps(final_d_re, final_d_re), 
        _mm256_mul_ps(final_d_im, final_d_im)
    );

    //u = z/dz (complex division)
    __m256 u_re = _mm256_div_ps(_mm256_add_ps(_mm256_mul_ps(final_re, final_d_re), _mm256_mul_ps(final_im, final_d_im)), dlen2);
    __m256 u_im = _mm256_div_ps(_mm256_sub_ps(_mm256_mul_ps(final_im, final_d_re), _mm256_mul_ps(final_re, final_d_im)), dlen2);

    const __m256 u_len = _mm256_sqrt_ps(_mm256_add_ps(_mm256_mul_ps(u_re, u_re), _mm256_mul_ps(u_im, u_im)));

    u_re = _mm256_div_ps(u_re, u_len);
    u_im = _mm256_div_ps(u_im, u_len);

    const __m256 dot = _mm256_add_ps(_mm256_add_ps(_mm256_mul_ps(u_re, lx), _mm256_mul_ps(u_im, ly)), lheight);

    __m256 res = _mm256_max_ps(zero, _mm256_min_ps(_mm256_div_ps(dot, _mm256_add_ps(one, lheight)), one));

    res = _mm256_blendv_ps(zero, res, condition);

	_mm256_store_ps(mem_address, res);
}