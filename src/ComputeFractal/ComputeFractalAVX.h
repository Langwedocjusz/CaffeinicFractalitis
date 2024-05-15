#pragma once

#include <immintrin.h>

namespace ComputeFractalAVX{
    //Uses avx instructions, stores (smoothed) iteration counts required
    //to reach bailout radius of eight positions at target address
    void Mandelbrot(float* mem_address, __m256  x, __m256 y);
    
    //Uses avx instructions, stores dot product of Mandelbrot potential 
    //gradient with a constant vector at target address
    //To-do: Fix artefacts in the shape of black lines along potential level sets
    void MandelbrotLight(float* mem_address, __m256 x, __m256 y);
}
