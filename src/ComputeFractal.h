#pragma once

#include <xmmintrin.h>
#include <smmintrin.h>

namespace ComputeFractal{
    //Returns (smoothed) iteration count required to reach a bailout radius
    float Mandelbrot(float c_re, float c_im);

    //Uses sse instructions, stores (smoothed) iteration counts required
    //to reach bailout radius of four positions at target address
    void Mandelbrot(float* mem_address, __m128  x, __m128 y);
};
