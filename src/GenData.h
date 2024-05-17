#pragma once

#include <vector>
#include <thread>
#include <cstdint>
#include <optional>
#include <iostream>

#include "AlignedAllocator.h"
#include "SimdType.h"
#include "ComputeFractal.h"

namespace GenData {

    struct ExecutionPolicy{
        SimdType Simd = SimdType::Scalar;
        std::optional<uint32_t> NumJobs = std::nullopt;
    };

    struct FrameParams{
        float MinX;
        float MaxX;
        float MinY;
        float MaxY;
        size_t Width;
        size_t Height;
    };

	void GenerateFractal(AlignedVector<float>& data, GenFunction f, FrameParams p, ExecutionPolicy e);
}
