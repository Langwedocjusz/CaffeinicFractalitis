#pragma once

#include <vector>
#include <string>

namespace SaveImage {
	void ColorAndSave(std::vector<float>& data, 
                      const std::string& filename,
                      size_t width, size_t height);
}
