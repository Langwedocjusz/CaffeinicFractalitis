#pragma once

#include <cstdint>
#include <vector>
#include <string>
#include <thread>
#include <optional>
#include <functional>

#include "AlignedAllocator.h"

namespace Image {
	struct Pixel {
		uint8_t r;
		uint8_t g;
		uint8_t b;
	};

	struct Pixelf {
		float r;
		float g;
		float b;
	};

	typedef std::function<Pixel(float)> ColoringFn;

	enum class ImageColoring{
		None,
		IterToColorIQ,
		NormedGrayscale,
		ColorHSV
	};

	ColoringFn GetColoringFunction(ImageColoring c);

	//Converts float values to grayscale pixels
	//assuming they were already normalized to [0,1]
	Pixel NormedGrayscale(float value);

	//Convers float values to colorful pixels
	//Hue is periodic function of the value
	//Brightness is monotonic mapping of [0, inf) onto [0, 1)
	//Saturation is constant
	Pixel ColorHSV(float value);

	//Conversion of iteration count to rgb color
	//Based on one written by Inigo Quilez and used for example here:
	//https://www.shadertoy.com/view/MltXz2
	Pixel IterToColorIQ(float iter_count);

	struct ImageInfo{
		uint32_t Width;
		uint32_t Height;
		std::string Name;
	};

	void SaveImage(std::vector<Pixel>& image, ImageInfo info);

	void ColorAndSave(AlignedVector<float>&data, ColoringFn f, ImageInfo info, std::optional<uint32_t> num_jobs = std::nullopt);
}
