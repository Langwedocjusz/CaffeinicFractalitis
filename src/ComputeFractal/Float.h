#pragma once

#include "SimdType.h"

#include <xmmintrin.h>
#include <smmintrin.h>
#include <immintrin.h>

template <SimdType T> struct Float{};

template <>
struct Float<SimdType::Scalar>{
    using enum SimdType;
    typedef float ValueType;

    float Value;

    void operator=(const float& value)
    {
        Value = value;
    }

    Float<Scalar> operator+(const Float<Scalar>& other)
    {
        return Float<Scalar>{Value + other.Value};
    }

    Float<Scalar> operator-(const Float<Scalar>& other)
    {
        return Float<Scalar>{Value - other.Value};
    }

    Float<Scalar> operator/(const Float<Scalar>& other)
    {
        return Float<Scalar>{Value / other.Value};
    }

    static Float<SimdType::Scalar> sqrt(Float<SimdType::Scalar> x)
    {
        return Float<SimdType::Scalar>{std::sqrt(x.Value)};
    }
};

static Float<SimdType::Scalar> operator+(float x, const Float<SimdType::Scalar>& X)
{
    return Float<SimdType::Scalar>{x + X.Value};
}

static Float<SimdType::Scalar> operator*(const Float<SimdType::Scalar>& lhs, const Float<SimdType::Scalar>& rhs)
{
    return Float<SimdType::Scalar>{lhs.Value * rhs.Value};
}

static Float<SimdType::Scalar> operator*(float x, const Float<SimdType::Scalar>& X)
{
    return Float<SimdType::Scalar>{x * X.Value};
}

static Float<SimdType::Scalar> operator+(const Float<SimdType::Scalar>& X, float x)
{
    return Float<SimdType::Scalar>{x + X.Value};
}

template <>
struct Float<SimdType::SSE>{
    using enum SimdType;
    typedef __m128 ValueType;

    __m128 Value;

    void operator=(const float& value)
    {
        Value = _mm_set1_ps(value);
    }

    Float<SSE> operator+(const Float<SSE>& other)
    {
        return Float<SSE>{
            _mm_add_ps(Value, other.Value)
        };
    }

    Float<SSE> operator-(const Float<SSE>& other)
    {
        return Float<SSE>{
            _mm_sub_ps(Value, other.Value)
        };
    }

    Float<SSE> operator/(const Float<SSE>& other)
    {
        return Float<SSE>{
            _mm_div_ps(Value, other.Value)
        };
    }

    static Float<SSE> sqrt(Float<SSE> x)
    {
        return Float<SSE>{
            _mm_sqrt_ps(x.Value)
        };
    }

    static Float<SSE> blend(Float<SSE> x, Float<SSE> y, ValueType condition)
    {
        return Float<SSE>{
            _mm_blendv_ps(x.Value, y.Value, condition)
        };
    }
};

static Float<SimdType::SSE> operator*(const Float<SimdType::SSE>& lhs, const Float<SimdType::SSE>& rhs)
{
    return Float<SimdType::SSE>{
        _mm_mul_ps(lhs.Value, rhs.Value)
    };
}

static Float<SimdType::SSE> operator*(float x, const Float<SimdType::SSE>& X)
{
    return Float<SimdType::SSE>{
        _mm_mul_ps(X.Value, _mm_set1_ps(x))
    };
}

static Float<SimdType::SSE> operator+(float x, const Float<SimdType::SSE>& X)
{
    return Float<SimdType::SSE>{
        _mm_add_ps(X.Value, _mm_set1_ps(x))
    };
}

static Float<SimdType::SSE> operator+(const Float<SimdType::SSE>& X, float x)
{
    return Float<SimdType::SSE>{
        _mm_add_ps(X.Value, _mm_set1_ps(x))
    };
}

template <>
struct Float<SimdType::AVX>{
    using enum SimdType;
    typedef __m256 ValueType;

    __m256 Value;

    void operator=(const float& value)
    {
        Value = _mm256_set1_ps(value);
    }

    Float<AVX> operator+(const Float<AVX>& other)
    {
        return Float<AVX>{
            _mm256_add_ps(Value, other.Value)
        };
    }

    Float<AVX> operator-(const Float<AVX>& other)
    {
        return Float<AVX>{
            _mm256_sub_ps(Value, other.Value)
        };
    }

    Float<AVX> operator/(const Float<AVX>& other)
    {
        return Float<AVX>{
            _mm256_div_ps(Value, other.Value)
        };
    }

    static Float<AVX> sqrt(Float<AVX> x)
    {
        return Float<AVX>{
            _mm256_sqrt_ps(x.Value)
        };
    }

    static Float<AVX> blend(Float<AVX> x, Float<AVX> y, ValueType condition)
    {
        return Float<AVX>{
            _mm256_blendv_ps(x.Value, y.Value, condition)
        };
    }
};

static Float<SimdType::AVX> operator*(const Float<SimdType::AVX>& lhs, const Float<SimdType::AVX>& rhs)
{
    return Float<SimdType::AVX>{
        _mm256_mul_ps(lhs.Value, rhs.Value)
    };
}

static Float<SimdType::AVX> operator*(float x, const Float<SimdType::AVX>& X)
{
    return Float<SimdType::AVX>{
        _mm256_mul_ps(X.Value, _mm256_set1_ps(x))
    };
}

static Float<SimdType::AVX> operator+(float x, const Float<SimdType::AVX>& X)
{
    return Float<SimdType::AVX>{
        _mm256_add_ps(X.Value, _mm256_set1_ps(x))
    };
}

static Float<SimdType::AVX> operator+(const Float<SimdType::AVX>& X, float x)
{
    return Float<SimdType::AVX>{
        _mm256_add_ps(X.Value, _mm256_set1_ps(x))
    };
}