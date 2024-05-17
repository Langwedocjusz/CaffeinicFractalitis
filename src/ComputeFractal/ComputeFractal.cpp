#include "ComputeFractal.h"

#include "Mandelbrot.h"
#include "MandelbrotLight.h"

#include <map>

GenFunction GetGeneratingFunction(Generator g)
{
    using namespace ComputeFractal;

    std::map<Generator, GenFunction> gen_functions{
        {Generator::Mandelbrot,      {Mandelbrot,      MandelbrotSSE,      MandelbrotAVX}},
        {Generator::MandelbrotLight, {MandelbrotLight, MandelbrotLightSSE, MandelbrotLightAVX}},
    };

    return gen_functions[g];
}



