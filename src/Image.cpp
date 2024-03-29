#include "Image.h"

#define STBI_MSC_SECURE_CRT
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include <cmath>
#include <array>

static Image::Pixelf hsv_to_rgb(float h, float s, float v);

void Image::SaveImage(std::vector<Pixel>& image,
	                  const std::string& filename,
	                  size_t width, size_t height)
{
	const size_t channel_nr = 3;

	stbi_write_png(filename.c_str(), width, height, channel_nr, &image[0], width * sizeof(Pixel));
}

Image::Pixel Image::NormedGrayscale(float value)
{
	const uint8_t v = static_cast<uint8_t>(255.0f * value);

	return Image::Pixel{ v, v, v };
}

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
