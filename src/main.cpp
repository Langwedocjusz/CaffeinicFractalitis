#include "Timer.h"

#include "GenData.h"
#include "Image.h"

#include "ComputeFractal.h"
#include "ComputeFractalSSE.h"

int main(){
    using GenData::ExecutionPolicy;

	constexpr size_t width  = 1024;
	constexpr size_t height = 1024;

	std::vector<float> data;
	data.resize(width * height);

    float half_ext = 1.0f;

    for (auto i=0; i<50; i++)
    {
        const float center_x = -0.74f;
        const float center_y = 0.1f;

        const GenData::FrameParams params{
            .MinX   = center_x - half_ext,
            .MaxX   = center_x + half_ext,
            .MinY   = center_y - half_ext,
            .MaxY   = center_y + half_ext,
            .Width  = width, 
            .Height = height,
        };

        const std::string filename = std::to_string(i) + ".png";

        {
            Timer we("Generating the fractal");
            GenData::GenerateFractal<ExecutionPolicy::SSE>(data, ComputeFractalSSE::MandelbrotLight, params);
        }

        {
            Timer we("Coloring and saving the image");
            Image::ColorAndSave(data, Image::NormedGrayscale, filename, width, height);
        }

        half_ext *= 0.9f;
    }
}
