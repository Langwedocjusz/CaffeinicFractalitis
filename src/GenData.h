#pragma once

#include <vector>

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

            for (size_t i = start; i < end; i++)
            {
                const float x = getX(i); 
                const float y = getY(i);
                
                data[i] = f(x, y);
            }
        };

#ifdef USE_MULTITHREADING
	    const size_t num_threads = std::thread::hardware_concurrency();
	    const size_t total = data.size();

	    std::vector<std::thread> threads;

	    for (size_t i = 0; i < num_threads; i++)
	    {
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
