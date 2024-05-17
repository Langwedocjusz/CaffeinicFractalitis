#pragma once

#include <xmmintrin.h>
#include <smmintrin.h>
#include <immintrin.h>

#include <functional>

enum class Generator{
    Mandelbrot, 
    MandelbrotLight
};

typedef std::function<float(float, float)> ScalarFunction;
typedef std::function<void(float*, __m128, __m128)> SSEFunction;
typedef std::function<void(float*, __m256, __m256)> AVXFunction;

struct GenFunction{
    ScalarFunction Scalar;
    SSEFunction SSE;
    AVXFunction AVX;
};

GenFunction GetGeneratingFunction(Generator g);