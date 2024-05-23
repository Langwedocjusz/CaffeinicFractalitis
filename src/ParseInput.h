#pragma once

#include <optional>
#include <string>
#include <string_view>
#include <cstdint>

#include "SimdType.h"

#include "ComputeFractal.h"
#include "Image.h"

struct ProgramArgs{
    uint32_t Width;
    uint32_t Height;
    uint32_t NumFrames;

    float CenterX;
    float CenterY;
    float InitialWidth;
    float ZoomSpeed;

    FractalGenerator Generator;
    Image::ImageColoring Coloring;

    std::optional<uint32_t> NumJobs;
    std::optional<SimdType> Simd;
    
    std::optional<std::string> ExitMessage;
};

ProgramArgs ParseInput(int argc, char* argv[]);