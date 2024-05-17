#pragma once

#include <xmmintrin.h>
#include <smmintrin.h>
#include <immintrin.h>

namespace ComputeFractal{
    //Returns (smoothed) iteration count required to reach a bailout radius
    //Based on this article by Inigo Quilez:
    //https://iquilezles.org/articles/msetsmooth/
    float Mandelbrot(float x, float y);
    //Same as above, but uses SSE instrucions
    void MandelbrotSSE(float* mem_address, __m128  x, __m128 y);
    //Same as above, but uses AVX instrucions
    void MandelbrotAVX(float* mem_address, __m256  x, __m256 y);
}