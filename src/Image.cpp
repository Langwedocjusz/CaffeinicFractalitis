#include "Image.h"

#define STBI_MSC_SECURE_CRT
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include <cmath>
#include <array>
#include <map>

Image::ColoringFn Image::GetColoringFunction(ImageColoring c)
{
	auto ReturnBlack = [](float){return Pixel{0,0,0};};

	const std::map<ImageColoring, ColoringFn> coloring_functions{
		{ImageColoring::None,            ReturnBlack},
		{ImageColoring::IterToColorIQ,   IterToColorIQ},
		{ImageColoring::NormedGrayscale, NormedGrayscale},
		{ImageColoring::ColorHSV,        ColorHSV}
	};

	return coloring_functions.at(c);
}

void Image::SaveImage(std::vector<Pixel>& image, ImageInfo info)
{
	const size_t channel_nr = 3;

	stbi_write_png(info.Name.c_str(), info.Width, info.Height, channel_nr, &image[0], info.Width * sizeof(Pixel));
}

void Image::ColorAndSave(AlignedVector<float>&data, ColoringFn f, ImageInfo info, std::optional<uint32_t> num_jobs)
{
	std::vector<Pixel> image(data.size());

	auto ColorPixels = [&](size_t start, size_t end)
	{
		for (size_t i = start; i < end; i++)
		{
			image[i] = f(data[i]);
		}
	};

	const size_t num_threads = [&](){
        if (num_jobs.has_value())
            return num_jobs.value();
        else
            return std::thread::hardware_concurrency();
    }();

    if (num_threads > 1)
    {
	    const size_t total = image.size();
		std::vector<std::thread> threads;

		for (size_t i = 0; i < num_threads; i++)
		{
			const size_t start = i * total / num_threads;
			const size_t end = (i + 1) * total / num_threads;
			threads.push_back(std::thread(ColorPixels, start, end));
		}
		
		for (auto& thread : threads)
			thread.join();
    }
    else
    {
        ColorPixels(0, image.size());
    }

	SaveImage(image, info);
}

Image::Pixel Image::NormedGrayscale(float value)
{
	const uint8_t v = static_cast<uint8_t>(255.0f * value);

	return Image::Pixel{ v, v, v };
}

Image::Pixel Image::IterToColorIQ(float iter_count)
{
	const float freq = 2.0f*0.075f;
	const std::array<float, 3> phase{3.0f + 0.0f, 3.0f + 0.6f, 3.0f + 1.0f};

	auto processColor = [=](uint32_t id){
		const float x = 0.5f + 0.5f*std::cos(freq*iter_count + phase[id]);

		return static_cast<uint32_t>(255.0f * x);
	};

	const uint8_t r = processColor(0);
	const uint8_t g = processColor(1);
	const uint8_t b = processColor(2);

	return Image::Pixel{r,g,b};
}

static Image::Pixelf hsv_to_rgb(float h, float s, float v);

Image::Pixel Image::ColorHSV(float value)
{
	auto normalize = [](float x)
	{
		const float lorentz = 1.0f - 1.0f / (1.0f + x * x);
		return lorentz * lorentz * lorentz;
	};

	const float h = 3.6 * std::fmod(value, 100.0f);
	const float s = 0.5;
	const float v = normalize(value);

	const Image::Pixelf rgb = hsv_to_rgb(h, s, v);

	const uint8_t r = static_cast<uint8_t>(255.0f * rgb.r);
	const uint8_t g = static_cast<uint8_t>(255.0f * rgb.g);
	const uint8_t b = static_cast<uint8_t>(255.0f * rgb.b);

	return Image::Pixel{ r, g, b };
}

static Image::Pixelf hsv_to_rgb(float h, float s, float v)
{
	using namespace Image;

	const float c = v * s;
	const float h1 = (h != 360.0f) ? (h / 60.0f) : 0.0f;
	const float x = c * (1.0f - std::abs(std::fmod(h1, 2.0f) - 1.0f));
	const float m = v - c;

	const uint32_t sixth_num = static_cast<uint32_t>(std::floor(h1));

	const std::array<Pixelf, 6> outputs{
		Pixelf{c, x, 0.0f},
		Pixelf{x, c, 0.0f},
		Pixelf{0.0f, c, x},
		Pixelf{0.0f, x, c},
		Pixelf{x, 0.0f, c},
		Pixelf{c, 0.0f, x}
	};

	Pixelf res = (sixth_num < 6) ? outputs.at(sixth_num) : Pixelf{ 0.0f, 0.0f, 0.0f };

	res.r += m;
	res.g += m;
	res.b += m;

	return res;
}
