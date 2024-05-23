#include "ComputeFractal.h"

#include "SmoothIter.h"
#include "Gradient.h"

#include <map>

GenFunction GetGeneratingFunction(FractalGenerator g)
{
    using namespace ComputeFractal;

    auto ReturnZero = [](float, float){return 0.0f;};
    auto DoNothingSSE = [](float*, __m128, __m128){};
    auto DoNothingAVX = [](float*, __m256, __m256){};

    const std::map<FractalGenerator, GenFunction> gen_functions{
        {FractalGenerator::None,       {ReturnZero, DoNothingSSE, DoNothingAVX}},
        {FractalGenerator::SmoothIter, {SmoothIter, SmoothIterSSE, SmoothIterAVX}},
        {FractalGenerator::Gradient,   {Gradient,   GradientSSE,   GradientAVX}},
    };

    return gen_functions.at(g);
}



