#pragma once

#include <chrono>
#include <string>

class Timer {
public:
	Timer(const std::string& msg);
	~Timer();

private:
	std::chrono::time_point<std::chrono::high_resolution_clock> m_Start;
	std::string m_Message;
};