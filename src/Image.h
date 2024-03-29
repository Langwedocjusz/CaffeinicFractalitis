#pragma once

#include <cstdint>
#include <vector>
#include <string>

#ifdef USE_MULTITHREADING
#include <thread>
#endif

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

	//Converts float values to grayscale pixels
	//assuming they were already normalized to [0,1]
	Pixel NormedGrayscale(float value);
	//Convers float values to colorful pixels
	//Hue is periodic function of the value
	//Brightness is monotonic mapping of [0, inf) onto [0, 1)
	//Saturation is constant
	Pixel ColorHSV(float value);

	void SaveImage(std::vector<Pixel>& image,
		const std::string& filename,
		size_t width, size_t height);

	template<typename Fn>
	void ColorAndSave(std::vector<float>& data,
		Fn coloring_function,
		const std::string& filename,
		size_t width, size_t height)
	{
		std::vector<Pixel> image;
		image.resize(width * height);

		auto ColorPixels = [&](size_t start, size_t end)
		{
			for (size_t i = start; i < end; i++)
			{
				image[i] = coloring_function(data[i]);
			}
		};

#ifdef USE_MULTITHREADING
		const size_t num_threads = std::thread::hardware_concurrency();
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
#else
		ColorPixels(0, image.size());
#endif
		SaveImage(image, filename, width, height);
	}
}
