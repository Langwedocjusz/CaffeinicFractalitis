#include "ComputeFractal.h"

#include <cmath>

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

float ComputeFractal::MandelbrotLight(float x, float y)
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
