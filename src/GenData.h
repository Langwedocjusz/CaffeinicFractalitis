#pragma once

#include <vector>

namespace GenData {
    struct Params{
        float min_x;
        float max_x;
        float min_y;
        float max_y;
        size_t width;
        size_t height;
    };

	void GenerateFractal(std::vector<float>& data, Params p);
}
