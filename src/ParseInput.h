#pragma once

#include <optional>
#include <string>
#include <string_view>
#include <cstdint>

#include "SimdType.h"

struct ProgramArgs{
    std::optional<uint32_t> NumJobs;
    std::optional<std::string> ExitMessage;
    std::optional<SimdType> Simd;
};

ProgramArgs ParseInput(int argc, char* argv[]);