#pragma once

namespace ComputeFractal{
    //Returns (smoothed) iteration count required to reach a bailout radius
    float Mandelbrot(float x, float y);

    //Returns dot product of Mandelbrot potential gradient with a constant vector
    float MandelbrotLight(float x, float y);
};
