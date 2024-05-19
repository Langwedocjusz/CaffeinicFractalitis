#include "Timer.h"

#include "ComputeFractal.h"
#include "GenData.h"
#include "Image.h"

#include "ParseInput.h"

int main(int argc, char* argv[])
{
    constexpr size_t width  = 1024;
	constexpr size_t height = 1024;
    constexpr size_t num_frames = 100;
    constexpr float center_x = -0.745f;
    constexpr float center_y = 0.1f;

    Generator gen_choice = Generator::Mandelbrot;
    const auto gen_function = GetGeneratingFunction(gen_choice);

    ProgramArgs args = ParseInput(argc, argv);
    
    if (args.ExitMessage.has_value())
    {
        std::cerr << args.ExitMessage.value() << '\n';
        return -1;
    }

    AlignedVector<float> data(width*height);

    SimdType simd_type = args.Simd.has_value()
                       ? args.Simd.value()
                       : SimdType::AVX;

    float half_ext = 1.0f;

    for (auto i=0; i<num_frames; i++)
    {
        const GenData::ExecutionPolicy exec_policy{
            .Simd = simd_type,
            .NumJobs = args.NumJobs
        };

        const GenData::FrameParams params{
            .MinX   = center_x - half_ext,
            .MaxX   = center_x + half_ext,
            .MinY   = center_y - half_ext,
            .MaxY   = center_y + half_ext,
            .Width  = width, 
            .Height = height
        };

        const Image::ImageInfo info{
            .Width = width,
            .Height = height,
            .Name = std::to_string(i) + ".png"
        };

        {
            Timer we("Generating the fractal");
            
            GenData::GenerateFractal(data, gen_function, params, exec_policy);
        }

        {
            Timer we("Coloring and saving the image");

            Image::ColorAndSave(data, Image::IterToColorIQ, info, args.NumJobs);
        }
 
        half_ext *= 0.9f;
    }
}