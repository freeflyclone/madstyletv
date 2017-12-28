#pragma region CPL License
/*
Nuclex Native Framework
Copyright (C) 2002-2012 Nuclex Development Labs

This library is free software; you can redistribute it and/or
modify it under the terms of the IBM Common Public License as
published by the IBM Corporation; either version 1.0 of the
License, or (at your option) any later version.

This library is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
IBM Common Public License for more details.

You should have received a copy of the IBM Common Public
License along with this library
*/
#pragma endregion // CPL License

// If the library is compiled as a DLL, this ensures symbols are exported
#define NUCLEX_GAME_SOURCE 1

#include "Timing/WindowsClock.h"
#include <limits>
#include <stdexcept>
#include <cassert>

// Don't use WIN32_LEAN_AND_MEAN in this case.
#include <Windows.h>

// Yes, Microsoft, a global define named "max" was a fantastic idea.
#if defined(max)
#undef max
#endif

namespace {

  // ------------------------------------------------------------------------------------------- //

  /// <summary>Determines the highest possible resolution the timer supports</summary>
  /// <returns>The smallest interval the timer can step by in milliseconds</returns>
  std::uint32_t getHighestPossibleTimerResolution() {
    TIMECAPS timeCaps;

    MMRESULT result = ::timeGetDevCaps(&timeCaps, sizeof(timeCaps));
    if(result != MMSYSERR_NOERROR) {
      throw std::runtime_error("Could not query timer capabilities");
    }

    return timeCaps.wPeriodMin;
  }

  // ------------------------------------------------------------------------------------------- //

  /// <summary>Configures the timer to at least the specified resolution</summary>
  /// <param name="resolution">Resolution the timer will be configured to</param>
  void enterTimerResolution(std::uint32_t resolution) {
    MMRESULT result = ::timeBeginPeriod(resolution);
    if(result != MMSYSERR_NOERROR) {
      throw std::runtime_error("Could not increase timer resolution");
    }
  }

  // ------------------------------------------------------------------------------------------- //

  /// <summary>Restores the previous resolution of the timer</summary>
  /// <param name="resolution">Resolution from which the timer will be restored</param>
  void leaveTimerResolution(std::uint32_t resolution) {
    MMRESULT result = ::timeEndPeriod(resolution);
    if(result != MMSYSERR_NOERROR) {
      throw std::runtime_error("Could not restore timer resolution");
    }
  }

  // ------------------------------------------------------------------------------------------- //

} // anonymous namespace

namespace Nuclex { namespace Game { namespace Timing {

  // ------------------------------------------------------------------------------------------- //

  WindowsClock::WindowsClock() :
    timerResolution(getHighestPossibleTimerResolution()) {

    // Windows desktop applications can increase the timer resolution at will
    enterTimerResolution(this->timerResolution);

    bool performanceCounterAvailable = TryGetCountFrequency(this->performanceCounterFrequency);
    if(performanceCounterAvailable) {
      std::uint64_t currentCounts = GetCurrentCounts();;
      this->performanceCounterOffset = getTimeAsCounts() - currentCounts;
      this->previousCounts = currentCounts + this->performanceCounterOffset;
    } else {
      this->performanceCounterFrequency = 0;
    }
  }

  // ------------------------------------------------------------------------------------------- //

  WindowsClock::~WindowsClock() {
    leaveTimerResolution(this->timerResolution);
  }

  // ------------------------------------------------------------------------------------------- //

  bool WindowsClock::TryGetCountFrequency(std::uint64_t &countFrequency) {
    ::LARGE_INTEGER frequency;

    BOOL isSupported = ::QueryPerformanceFrequency(&frequency);
    if(isSupported == FALSE) {
      countFrequency = 0;
      return false;
    } else {
      countFrequency = frequency.QuadPart;
      return true;
    }
  }

  // ------------------------------------------------------------------------------------------- //

  std::uint64_t WindowsClock::GetCountFrequency() {
    std::uint64_t countFrequency;

    if(!TryGetCountFrequency(countFrequency)) {
      throw std::runtime_error("The system does not provide a performance counter");
    }

    return countFrequency;
  }

  // ------------------------------------------------------------------------------------------- //

  std::uint64_t WindowsClock::GetCurrentCounts() {
    ::LARGE_INTEGER counts;

    BOOL wasQueried = ::QueryPerformanceCounter(&counts);
    if(wasQueried == FALSE) {
      throw std::runtime_error("The performance counter could not be queried");
    }

    return counts.QuadPart;
  }

  // ------------------------------------------------------------------------------------------- //

  std::uint32_t WindowsClock::GetUptime() {
    return static_cast<std::uint32_t>(::timeGetTime());
  }

  // ------------------------------------------------------------------------------------------- //

  std::uint64_t WindowsClock::GetWraparoundTime() const {
    if(this->performanceCounterFrequency == 0) { // if 0, no performance counter is available
      return std::numeric_limits<std::uint32_t>::max();
    } else {
      return std::numeric_limits<std::uint64_t>::max();
    }
  }

  // ------------------------------------------------------------------------------------------- //

  std::uint64_t WindowsClock::GetTime() const {
    if(this->performanceCounterFrequency == 0) { // if 0, no performance counter is available
      return GetUptime();
    }

    // Normally we could just return the performance counter. But out in the wild are systems
    // with PCI-to-ISA bridges that make performance counters jump randomly by several seconds
    // and systems with broken BIOSes that cause counters to have different states across
    // CPU cores. We have to compensate for this.

    std::uint64_t currentCounts = GetCurrentCounts() + this->performanceCounterOffset;
    std::uint64_t timeAsCounts = getTimeAsCounts();

    // Calculate the difference between the time and the performance counter
    std::int64_t difference = (currentCounts - timeAsCounts);
    {
      std::int64_t wraparoundOffset =
        std::numeric_limits<std::uint32_t>::max() * this->performanceCounterFrequency / 1000;

      // Did the time wrap around? We need to recalculate the offset.
      if(difference > (wraparoundOffset / 2)) {
        difference -= wraparoundOffset;
        this->performanceCounterOffset += timeAsCounts - currentCounts;
      }
    }

    // If the performance counter goes backwards or is off by more than 3 times
    // the timer resolution Windows boasts of supporting, we assume that
    // the performance counter has leaped.
    bool leapOccurred;
    {
      std::int64_t allowedDeviationCounts =
        this->timerResolution * 3 * this->performanceCounterFrequency / 1000;

      leapOccurred =
        (currentCounts < this->previousCounts) || // Jumped backwards
        (abs(difference) > allowedDeviationCounts); // Jumped too far
    }

    // If the performance counter leaped, fall back to the time API (but make sure time never
    // goes backwards) and resynchronize the performance counter offset to the leap location.
    if(leapOccurred) {
      this->previousCounts = std::max(this->previousCounts, timeAsCounts);
      this->performanceCounterOffset += timeAsCounts - currentCounts;
    } else { // Everything seems normal, we trust the performance counter
      this->previousCounts = currentCounts;
    }

    return this->previousCounts;
  }

  // ------------------------------------------------------------------------------------------- //

  std::uint64_t WindowsClock::GetFrequency() const {
    if(this->performanceCounterFrequency == 0) { // if 0, no performance counter is available
      return 1000; // The fallback timer reports time in millseconds
    } else {
      return this->performanceCounterFrequency;
    }
  }

  // ------------------------------------------------------------------------------------------- //

  std::uint64_t WindowsClock::getTimeAsCounts() const {
    using namespace std;
    assert(
      (this->performanceCounterFrequency != 0) &&
      "Performance counter must be available if this method is called"
    );

    return static_cast<std::uint64_t>(GetUptime()) * this->performanceCounterFrequency / 1000;
  }

  // ------------------------------------------------------------------------------------------- //

}}} // namespace Nuclex::Game::Timing
