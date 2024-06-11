#pragma once

#include <xmmintrin.h>
#include <smmintrin.h>
#include <immintrin.h>

enum class FractalGenerator{
    None,
    SmoothIter,
    Gradient
};

typedef float (*ScalarFunction)(float, float);
typedef void (*SSEFunction)(float*, __m128, __m128);
typedef void (*AVXFunction)(float*, __m256, __m256);

struct GenFunction{
    ScalarFunction Scalar;
    SSEFunction SSE;
    AVXFunction AVX;
};

GenFunction GetGeneratingFunction(FractalGenerator g);