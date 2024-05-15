#pragma once

#include <xmmintrin.h>
#include <smmintrin.h>

namespace ComputeFractalSSE{
    //Uses sse instructions, stores (smoothed) iteration counts required
    //to reach bailout radius of four positions at target address
    void Mandelbrot(float* mem_address, __m128  x, __m128 y);
    
    //Uses sse instructions, stores dot product of Mandelbrot potential 
    //gradient with a constant vector at target address
    void MandelbrotLight(float* mem_address, __m128 x, __m128 y);
}
