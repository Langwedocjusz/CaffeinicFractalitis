#include "GenFractal.h"
#include "SaveImage.h"

#include <cmath>

int main(){
	constexpr size_t width  = 4096;
	constexpr size_t height = 4096;

	std::vector<float> data;
	data.resize(width * height);

	GenFractal::GenerateFractal(data, width, height);
	SaveImage::ColorAndSave(data, width, height);
}