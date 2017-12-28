#ifndef XTIMER_H
#define XTIMER_H

#include <chrono>
#include <ctime>
#include <thread>

#include "xutils.h"

#include "Timing/SteppedTimer.h"
#include "Timing/ScaledTimer.h"

namespace {
	using namespace Nuclex::Game::Timing;
}

#ifdef _WIN32
#include <Windows.h>
class XTimer {

public:
	XTimer() {
		QueryPerformanceFrequency(&frequency);
		QueryPerformanceCounter(&epoch);
	}
	double Since() {
		QueryPerformanceCounter(&since);
		double diff = (double)(since.QuadPart - epoch.QuadPart) / (double)frequency.QuadPart;
		return diff;
	}
	double SinceLast() {
		QueryPerformanceCounter(&since);
		double diff = (double)(since.QuadPart - last.QuadPart) / (double)frequency.QuadPart;
		last = since;
		return diff;
	}

	LARGE_INTEGER epoch{}, since{}, last{};
	LARGE_INTEGER frequency{};

	SteppedTimer steppedTimer;
};
#else
class XTimer {
public:
	XTimer() {
		epoch = std::chrono::high_resolution_clock::now();
	};

	double Since() {
		since = std::chrono::high_resolution_clock::now();
		diff = since - epoch;
		return diff.count();
	}
	double SinceLast() {
		since = std::chrono::high_resolution_clock::now();
		diff = since - last;
		last = since;
		return diff.count();
	}

	std::chrono::time_point<std::chrono::high_resolution_clock> epoch,since,last;
	std::chrono::duration<double> diff;

};
#endif

#endif