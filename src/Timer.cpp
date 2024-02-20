#include "Timer.h"

#include <iostream>

Timer::Timer(const std::string& msg)
	: m_Message(msg)
{
	m_Start = std::chrono::high_resolution_clock::now();
}

Timer::~Timer()
{
	auto now = std::chrono::high_resolution_clock::now();

	std::cout << m_Message << " took "
		<< std::chrono::duration<float, std::milli>(now - m_Start).count() / 1000.0f
		<< "[s]\n";
}