#pragma once

#include <xmmintrin.h>
#include <smmintrin.h>
#include <immintrin.h>

namespace ComputeFractal{
    //Returns dot product of Mandelbrot potential gradient with a constant vector
    //Based on 'Normal map effect' technique from here:
    //https://www.math.univ-toulouse.fr/~cheritat/wiki-draw/index.php/Mandelbrot_set
    float Gradient(float x, float y);
    //Same as above, but uses SSE instructions
    void GradientSSE(float* mem_address, __m128 x, __m128 y);
    //Same as above, but uses AVX instrucions
    //To-do: Fix black, box-shaped artefacts
    void GradientAVX(float* mem_address, __m256 x, __m256 y);
}