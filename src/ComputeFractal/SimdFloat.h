#pragma once

#include "SimdType.h"

#include <cmath>

#include <xmmintrin.h>
#include <smmintrin.h>
#include <immintrin.h>

template <SimdType T> struct SimdFloat{};

template <>
struct SimdFloat<SimdType::Scalar>{
    using enum SimdType;
    typedef float ValueType;

    float Value;

    void operator=(const float& value)
    {
        Value = value;
    }

    SimdFloat<Scalar> operator+(const SimdFloat<Scalar>& other)
    {
        return SimdFloat<Scalar>{Value + other.Value};
    }

    SimdFloat<Scalar> operator-(const SimdFloat<Scalar>& other)
    {
        return SimdFloat<Scalar>{Value - other.Value};
    }

    SimdFloat<Scalar> operator/(const SimdFloat<Scalar>& other)
    {
        return SimdFloat<Scalar>{Value / other.Value};
    }

    static SimdFloat<SimdType::Scalar> sqrt(SimdFloat<SimdType::Scalar> x)
    {
        return SimdFloat<SimdType::Scalar>{std::sqrt(x.Value)};
    }
};

inline SimdFloat<SimdType::Scalar> operator+(float x, const SimdFloat<SimdType::Scalar>& X)
{
    return SimdFloat<SimdType::Scalar>{x + X.Value};
}

inline SimdFloat<SimdType::Scalar> operator*(const SimdFloat<SimdType::Scalar>& lhs, const SimdFloat<SimdType::Scalar>& rhs)
{
    return SimdFloat<SimdType::Scalar>{lhs.Value * rhs.Value};
}

inline SimdFloat<SimdType::Scalar> operator*(float x, const SimdFloat<SimdType::Scalar>& X)
{
    return SimdFloat<SimdType::Scalar>{x * X.Value};
}

inline SimdFloat<SimdType::Scalar> operator+(const SimdFloat<SimdType::Scalar>& X, float x)
{
    return SimdFloat<SimdType::Scalar>{x + X.Value};
}

template <>
struct SimdFloat<SimdType::SSE>{
    using enum SimdType;
    typedef __m128 ValueType;

    __m128 Value;

    void operator=(const float& value)
    {
        Value = _mm_set1_ps(value);
    }

    SimdFloat<SSE> operator+(const SimdFloat<SSE>& other)
    {
        return SimdFloat<SSE>{
            _mm_add_ps(Value, other.Value)
        };
    }

    SimdFloat<SSE> operator-(const SimdFloat<SSE>& other)
    {
        return SimdFloat<SSE>{
            _mm_sub_ps(Value, other.Value)
        };
    }

    SimdFloat<SSE> operator/(const SimdFloat<SSE>& other)
    {
        return SimdFloat<SSE>{
            _mm_div_ps(Value, other.Value)
        };
    }

    static SimdFloat<SSE> sqrt(SimdFloat<SSE> x)
    {
        return SimdFloat<SSE>{
            _mm_sqrt_ps(x.Value)
        };
    }

    static SimdFloat<SSE> blend(SimdFloat<SSE> x, SimdFloat<SSE> y, ValueType condition)
    {
        return SimdFloat<SSE>{
            _mm_blendv_ps(x.Value, y.Value, condition)
        };
    }
};

inline SimdFloat<SimdType::SSE> operator*(const SimdFloat<SimdType::SSE>& lhs, const SimdFloat<SimdType::SSE>& rhs)
{
    return SimdFloat<SimdType::SSE>{
        _mm_mul_ps(lhs.Value, rhs.Value)
    };
}

inline SimdFloat<SimdType::SSE> operator*(float x, const SimdFloat<SimdType::SSE>& X)
{
    return SimdFloat<SimdType::SSE>{
        _mm_mul_ps(X.Value, _mm_set1_ps(x))
    };
}

inline SimdFloat<SimdType::SSE> operator+(float x, const SimdFloat<SimdType::SSE>& X)
{
    return SimdFloat<SimdType::SSE>{
        _mm_add_ps(X.Value, _mm_set1_ps(x))
    };
}

inline SimdFloat<SimdType::SSE> operator+(const SimdFloat<SimdType::SSE>& X, float x)
{
    return SimdFloat<SimdType::SSE>{
        _mm_add_ps(X.Value, _mm_set1_ps(x))
    };
}

template <>
struct SimdFloat<SimdType::AVX>{
    using enum SimdType;
    typedef __m256 ValueType;

    __m256 Value;

    void operator=(const float& value)
    {
        Value = _mm256_set1_ps(value);
    }

    SimdFloat<AVX> operator+(const SimdFloat<AVX>& other)
    {
        return SimdFloat<AVX>{
            _mm256_add_ps(Value, other.Value)
        };
    }

    SimdFloat<AVX> operator-(const SimdFloat<AVX>& other)
    {
        return SimdFloat<AVX>{
            _mm256_sub_ps(Value, other.Value)
        };
    }

    SimdFloat<AVX> operator/(const SimdFloat<AVX>& other)
    {
        return SimdFloat<AVX>{
            _mm256_div_ps(Value, other.Value)
        };
    }

    static SimdFloat<AVX> sqrt(SimdFloat<AVX> x)
    {
        return SimdFloat<AVX>{
            _mm256_sqrt_ps(x.Value)
        };
    }

    static SimdFloat<AVX> blend(SimdFloat<AVX> x, SimdFloat<AVX> y, ValueType condition)
    {
        return SimdFloat<AVX>{
            _mm256_blendv_ps(x.Value, y.Value, condition)
        };
    }
};

inline SimdFloat<SimdType::AVX> operator*(const SimdFloat<SimdType::AVX>& lhs, const SimdFloat<SimdType::AVX>& rhs)
{
    return SimdFloat<SimdType::AVX>{
        _mm256_mul_ps(lhs.Value, rhs.Value)
    };
}

inline SimdFloat<SimdType::AVX> operator*(float x, const SimdFloat<SimdType::AVX>& X)
{
    return SimdFloat<SimdType::AVX>{
        _mm256_mul_ps(X.Value, _mm256_set1_ps(x))
    };
}

inline SimdFloat<SimdType::AVX> operator+(float x, const SimdFloat<SimdType::AVX>& X)
{
    return SimdFloat<SimdType::AVX>{
        _mm256_add_ps(X.Value, _mm256_set1_ps(x))
    };
}

inline SimdFloat<SimdType::AVX> operator+(const SimdFloat<SimdType::AVX>& X, float x)
{
    return SimdFloat<SimdType::AVX>{
        _mm256_add_ps(X.Value, _mm256_set1_ps(x))
    };
}