#include "GenData.h"

#include <functional>

namespace GenData {

    typedef std::function<float(size_t)> CoordFunction;

    static void InnerLoop(AlignedVector<float>& data, GenFunction f,
        CoordFunction get_x, CoordFunction get_y,
        size_t start, size_t end,
        SimdType simd);

    void GenerateFractal(AlignedVector<float>& data, GenFunction f, FrameParams p, ExecutionPolicy e)
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

            InnerLoop(data, f, getX, getY, start, end, e.Simd);
        };

        const size_t num_threads = [&](){
            if (e.NumJobs.has_value())
                return e.NumJobs.value();
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


    static void InnerLoop(AlignedVector<float>& data, GenFunction f,
            CoordFunction get_x, CoordFunction get_y,
            size_t start, size_t end,
            SimdType simd)
    {
        using enum SimdType;

        auto IterateScalar = [&](size_t scalar_start, size_t scalar_end)
        {
            for (size_t i = scalar_start; i < scalar_end; i++)
            {
                const float x = get_x(i);
                const float y = get_y(i);
                data[i] = f.Scalar(x, y);
            }
        };

        //Utilities to find multiple of alignment that is closest to given value
        //from above and below:

        auto FindSmallerMultiple = [](size_t value, size_t alignment){
            const size_t mod = value % alignment;
            return value - mod;
        };

        auto FindLargerMultiple = [](size_t value, size_t alignment)
        {
            const size_t mod = value % alignment;
            return value - mod + alignment;
        };

        switch(simd)
        {
            case Scalar:
            {
                IterateScalar(start, end);
                break;
             }
            case SSE:
            {
                size_t vector_start = FindLargerMultiple(start, 4);
                size_t vector_end = FindSmallerMultiple(end, 4);

                IterateScalar(start, vector_start);

                for (size_t i = vector_start; i < vector_end; i += 4)
                {
                    __m128 x = _mm_set_ps(get_x(i + 3), get_x(i + 2), get_x(i + 1), get_x(i));
                    __m128 y = _mm_set_ps(get_y(i + 3), get_y(i + 2), get_y(i + 1), get_y(i));

                    float* mem_address = &data[i];
                    f.SSE(mem_address, x, y);
                }

                IterateScalar(vector_end, end);

                break;
            }
            case AVX:
            {
                size_t vector_start = FindLargerMultiple(start, 8);
                size_t vector_end = FindSmallerMultiple(end, 8);

                IterateScalar(start, vector_start);

                for (size_t i = start; i < end; i += 8)
                {
                    __m256 x = _mm256_set_ps(
                        get_x(i + 7), get_x(i + 6), get_x(i + 5), get_x(i + 4),
                        get_x(i + 3), get_x(i + 2), get_x(i + 1), get_x(i)
                    );

                    __m256 y = _mm256_set_ps(
                        get_y(i + 7), get_y(i + 6), get_y(i + 5), get_y(i + 4),
                        get_y(i + 3), get_y(i + 2), get_y(i + 1), get_y(i)
                    );

                    float* mem_address = &data[i];
                    f.AVX(mem_address, x, y);
                }

                IterateScalar(vector_end, end);

                break;
            }
        }
    }
}