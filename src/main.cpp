#include "Timer.h"

#include "ComputeFractal.h"
#include "GenData.h"
#include "Image.h"

#include "ParseInput.h"

int main(int argc, char* argv[])
{
    ProgramArgs args = ParseInput(argc, argv);

    if (args.ExitMessage.has_value())
    {
        std::cerr << args.ExitMessage.value() << '\n';
        return -1;
    }

    AlignedVector<float> data(args.Width*args.Height);

    auto gen_function = GetGeneratingFunction(args.Generator);
    auto coloring_fn = Image::GetColoringFunction(args.Coloring);

    SimdType simd_type = args.Simd.has_value()
                       ? args.Simd.value()
                       : SimdType::SSE;

    const float aspect_ratio = static_cast<float>(args.Height)/static_cast<float>(args.Width);

    float half_ext = 0.5f * args.InitialWidth;

    for (uint32_t i=0; i<args.NumFrames; i++)
    {
        const GenData::ExecutionPolicy exec_policy{
            .Simd = simd_type,
            .NumJobs = args.NumJobs
        };

        const GenData::FrameParams params{
            .MinX   = args.CenterX - half_ext,
            .MaxX   = args.CenterX + half_ext,
            .MinY   = args.CenterY - aspect_ratio*half_ext,
            .MaxY   = args.CenterY + aspect_ratio*half_ext,
            .Width  = args.Width,
            .Height = args.Height
        };

        const Image::ImageInfo info{
            .Width  = args.Width,
            .Height = args.Height,
            .Name   = std::to_string(i) + ".png"
        };

        {
            Timer we("Generating the fractal");

            GenData::GenerateFractal(data, gen_function, params, exec_policy);
        }

        {
            Timer we("Coloring and saving the image");

            Image::ColorAndSave(data, coloring_fn, info, args.NumJobs);
        }

        half_ext *= args.ZoomSpeed;
    }
}