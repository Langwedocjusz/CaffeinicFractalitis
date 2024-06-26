#pragma once

#include "SimdFloat.h"

template<SimdType T>
struct Complex{
    typedef SimdFloat<T>::ValueType ValueType;

    SimdFloat<T> Re;
    SimdFloat<T> Im;

    Complex(SimdFloat<T> re, SimdFloat<T> im)
        : Re(re), Im(im)
    {}

    Complex(float re, float im)
    {
        Re = re;
        Im = im;
    }

    Complex(ValueType re, ValueType im) requires(T != SimdType::Scalar)
    {
        Re.Value = re;
        Im.Value = im;
    }

    Complex operator+(const Complex& other)
    {
        return Complex{
            Re + other.Re, 
            Im + other.Im
        };
    }

    Complex operator*(const Complex& other)
    {
        return Complex{
            Re * other.Re - Im * other.Im, 
            Re * other.Im + Im * other.Re
        };
    }

    Complex operator/(const Complex& other)
    {
        const auto len2 = other.Re * other.Re + other.Im * other.Im;

        return Complex{
            (Re * other.Re + Im * other.Im)/len2, 
            (Im * other.Re - Re * other.Im)/len2
        };
    }

    void operator/=(SimdFloat<T> v)
    {
        Re = Re/v;
        Im = Im/v;
    }

    static SimdFloat<T> Len2(Complex z)
    {
        return z.Re * z.Re + z.Im * z.Im;
    }

    static SimdFloat<T> Len(Complex z)
    {
        return SimdFloat<T>::sqrt(Len2(z));
    }

    static SimdFloat<T> Dot(Complex x, Complex y)
    {
        return x.Re * y.Re + x.Im * y.Im;
    }

    static Complex Blend(Complex x, Complex y, ValueType condition)
    {
        return Complex{
            SimdFloat<T>::blend(x.Re, y.Re, condition),
            SimdFloat<T>::blend(x.Im, y.Im, condition),
        };
    }
};

template<SimdType T>
Complex<T> operator*(float x, const Complex<T>& z)
{
    return Complex{x * z.Re, x * z.Im};
}

template<SimdType T>
Complex<T> operator/(const Complex<T>& z, SimdFloat<T> x)
{
    return Complex{z.Re / x, z.Im / x};
}

template<SimdType T>
Complex<T> operator+(const Complex<T>& z, float x)
{
    return Complex{x + z.Re, z.Im};
}


