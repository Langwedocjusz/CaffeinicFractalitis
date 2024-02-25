#include "ComputeFractalSSE.h"
#include "GenDataSSE.h"
#include "SaveImage.h"

#include <cmath>

int main(){
	constexpr size_t width  = 1024;
	constexpr size_t height = 1024;

	std::vector<float> data;
	data.resize(width * height);

    float half_ext = 1.0f;

    for (auto i=0; i<50; i++)
    {
        const float center_x = -0.74f;
        const float center_y = 0.1f;

        const float min_x = center_x - half_ext;
        const float max_x = center_x + half_ext;
        const float min_y = center_y - half_ext;
        const float max_y = center_y + half_ext;

        GenData::Params params{
            min_x, max_x,
            min_y, max_y,
            width, height,
        };

        const std::string filename = std::to_string(i) + ".png";

	    GenData::GenerateFractal(data, ComputeFractal::MandelbrotLight, params);
	    SaveImage::ColorAndSave(data, filename, width, height);

        half_ext *= 0.9f;
    }
}
