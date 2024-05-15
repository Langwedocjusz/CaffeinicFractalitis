#pragma once

namespace ComputeFractal{
    //Returns (smoothed) iteration count required to reach a bailout radius
    //Based on this article by Inigo Quilez:
    //https://iquilezles.org/articles/msetsmooth/
    float Mandelbrot(float x, float y);

    //Returns dot product of Mandelbrot potential gradient with a constant vector
    //Based on 'Normal map effect' technique from here:
    //https://www.math.univ-toulouse.fr/~cheritat/wiki-draw/index.php/Mandelbrot_set
    float MandelbrotLight(float x, float y);
};
