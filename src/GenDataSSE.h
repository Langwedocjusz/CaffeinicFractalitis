#pragma once

#include "Timer.h"

#include <vector>
#include <xmmintrin.h>
#include <smmintrin.h>

#define USE_MULTITHREADING

#ifdef USE_MULTITHREADING
#include <thread>
#include <cstdint>
#endif

namespace GenData {

    struct Params{
        float min_x;
        float max_x;
        float min_y;
        float max_y;
        size_t width;
        size_t height;
    };
    
    template<typename Fn>
	void GenerateFractal(std::vector<float>& data, Fn f, Params p)
    {
	    Timer we("Generating the fractal");

        auto IterateImage = [&](size_t start, size_t end)
        {
            const float inv_width  = 1.0f/static_cast<float>(p.width);
            const float inv_height = 1.0f/static_cast<float>(p.height);
        
            const float extents_x = p.max_x - p.min_x;
            const float extents_y = p.max_y - p.min_y;
        	
            auto getX = [&](size_t id)
        	{
        		const float idx = static_cast<float>(id % p.width);
        		return extents_x * idx * inv_width + p.min_x;
        	};
        
        	auto getY = [&](size_t id)
        	{
        		const float idy = static_cast<float>(p.height - id / p.width);
        		return extents_y * idy * inv_height + p.min_y;
        	};

            for (size_t i = start; i < end; i += 4)
            {
                __m128  x, y;
                
                x = _mm_set_ps(getX(i + 3), getX(i + 2), getX(i + 1), getX(i));
                y = _mm_set_ps(getY(i + 3), getY(i + 2), getY(i + 1), getY(i));
                
                float* mem_address = &data[i];
                
                f(mem_address, x, y);
               }
        };

#ifdef USE_MULTITHREADING
	    const size_t num_threads = std::thread::hardware_concurrency();
	    const size_t total = data.size();

	    std::vector<std::thread> threads;

	    for (size_t i = 0; i < num_threads; i++)
	    {
		    //Start and end are currenlty assumed to both be
		    //divisible by 4 in the simd version
		    const size_t start =   i * total / num_threads;
		    const size_t end = (i+1) * total / num_threads;

		    threads.push_back(std::thread(IterateImage, start, end));
	    }

	    for (auto& thread : threads)
		    thread.join();
#else
	    IterateImage(0, data.size());
#endif
    }
}
