#pragma once

#include <immintrin.h>

namespace ComputeFractalAVX{
    //Uses avx instructions, stores (smoothed) iteration counts required
    //to reach bailout radius of eight positions at target address
    void Mandelbrot(float* mem_address, __m256  x, __m256 y);
    
    //Uses avx instructions, stores dot product of Mandelbrot potential 
    //gradient with a constant vector at target address
    //To-do: Currently there is a bug in the imlementation, which causes the
    //lower-left quarter to be darker then it should be.
    //To-do: There are also some artefacts in the shape of black lines
    void MandelbrotLight(float* mem_address, __m256 x, __m256 y);
}
