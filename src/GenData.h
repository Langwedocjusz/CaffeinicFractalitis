#pragma once

#include <vector>
#include <thread>
#include <cstdint>
#include <optional>

#include <xmmintrin.h>
#include <smmintrin.h>

namespace GenData {

    enum class ExecutionPolicy{
        Scalar,
        SSE
    };

    template <ExecutionPolicy E, typename Fn, typename Fx, typename Fy>
    typename std::enable_if<(E == ExecutionPolicy::Scalar), void>::type
    inner_loop(std::vector<float>& data, Fn f, Fx get_x, Fy get_y, size_t start, size_t end) 
    {
        for (size_t i = start; i < end; i++)
        {
            const float x = get_x(i); 
            const float y = get_y(i);
            data[i] = f(x, y);
        }
    }

    template <ExecutionPolicy E, typename Fn, typename Fx, typename Fy>
    typename std::enable_if<(E == ExecutionPolicy::SSE), void>::type
    inner_loop(std::vector<float>& data, Fn f, Fx get_x, Fy get_y, size_t start, size_t end) 
    {
        for (size_t i = start; i < end; i += 4)
        {
            __m128  x, y;

            x = _mm_set_ps(get_x(i + 3), get_x(i + 2), get_x(i + 1), get_x(i));
            y = _mm_set_ps(get_y(i + 3), get_y(i + 2), get_y(i + 1), get_y(i));

            float* mem_address = &data[i];
            f(mem_address, x, y);
        }
    }

    struct FrameParams{
        float MinX;
        float MaxX;
        float MinY;
        float MaxY;
        size_t Width;
        size_t Height;
    };

    template<ExecutionPolicy E, typename Fn>
	void GenerateFractal(std::vector<float>& data, Fn f, FrameParams p, std::optional<uint32_t> num_jobs = std::nullopt)
    {
        auto IterateImage = [&](size_t start, size_t end)
        {
            const float inv_width  = 1.0f/static_cast<float>(p.Width);
            const float inv_height = 1.0f/static_cast<float>(p.Height);
        
            const float extents_x = p.MaxX - p.MinX;
            const float extents_y = p.MaxY - p.MinY;
        	
            auto getX = [&](size_t id)
        	{
        		const float idx = static_cast<float>(id % p.Width);
        		return extents_x * idx * inv_width + p.MinX;
        	};
        
        	auto getY = [&](size_t id)
        	{
        		const float idy = static_cast<float>(p.Height - id / p.Width);
        		return extents_y * idy * inv_height + p.MinY;
        	};

            inner_loop<E>(data, f, getX, getY, start, end);
        };

        const size_t num_threads = [&](){
            if (num_jobs.has_value())
                return num_jobs.value();
            else
                return std::thread::hardware_concurrency();
        }();

        if (num_threads > 1)
        {
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
        }

        else
        {
            IterateImage(0, data.size());
        }
    }

}
