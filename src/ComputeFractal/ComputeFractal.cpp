#include "ComputeFractal.h"

#include "SmoothIter.h"
#include "Gradient.h"

#include <map>

GenFunction GetGeneratingFunction(Generator g)
{
    using namespace ComputeFractal;

    std::map<Generator, GenFunction> gen_functions{
        {Generator::SmoothIter, {SmoothIter, SmoothIterSSE, SmoothIterAVX}},
        {Generator::Gradient,   {Gradient,   GradientSSE,   GradientAVX}},
    };

    return gen_functions[g];
}



