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

#ifndef NUCLEX_GAME_TIMING_WINDOWSCLOCK_H
#define NUCLEX_GAME_TIMING_WINDOWSCLOCK_H

#include "Clock.h"

#include <cstddef>

namespace Nuclex { namespace Game { namespace Timing {

  // ------------------------------------------------------------------------------------------- //

  /// <summary>Uses the Windows API to obtain accurate timings on Microsoft systems</summary>
  /// <remarks>
  ///   This clock uses the performance counter API if available to obtain a very high
  ///   resolution timer (usually in the GHz range) on Windows system. If a performance
  ///   counter is not available, it will opt for timeGetTime(), which only offers
  ///   a resolution of 1 KHz.
  /// </remarks>
  class WindowsClock : public Clock {

    /// <summary>Initializes a new manually Windows clock</summary>
    public: WindowsClock();

    /// <summary>Destroys the clock</summary>
    public: ~WindowsClock();

    /// <summary>Tries to determine the performance counter frequency</summary>
    /// <param name="countFrequency">Receives the frequency of the performance counter</param>
    /// <returns>
    ///   True if the performance counter is available and its frequency was stored in
    ///   the provided variable, false otherwise.
    /// </returns>
    public: static bool TryGetCountFrequency(std::uint64_t &countFrequency);

    /// <summary>Returns the number of counts per second</summary>
    /// <returns>The total number of counts per second for the timer being used</returns>
    public: static std::uint64_t GetCountFrequency();

    /// <summary>Returns the number of counts that have passed</summary>
    /// <returns>The total number of counts that have passed since a defined time</returns>
    /// <remarks>
    ///   The &quot;defined time&quot; can be anything, typically it's either when the process
    ///   started or when the system was booted. The absolute value has no particular meaning
    ///   and should only be used to compare against an earlier value.
    /// </remarks>
    public: static std::uint64_t GetCurrentCounts();

    /// <summary>Retrieves the current system uptime in milliseconds</summary>
    /// <returns>The system uptime in milliseconds</returns>
    public: static std::uint32_t GetUptime();

    /// <summary>
    ///   Retrieves the maximum value the time can assume before wrapping around to 0 again
    /// </summary>
    /// <returns>The highest possible value GetTime() can return</returns>
    public: std::uint64_t GetWraparoundTime() const;

    /// <summary>Retrieves the clock's current time</summary>
    /// <returns>The number of ticks the clock has ticked to far</returns>
    /// <remarks>
    ///   There's no rule for what a clock's time should be relative to, only that it
    ///   keeps counting at a fixed pace. The clock could begin ticking when the system
    ///   boots, when the application starts or when the class is instantiated.
    /// </remarks>
    public: std::uint64_t GetTime() const;

    /// <summary>Retrieves the clock's tick frequency</summary>
    /// <returns>How often the clock will tick in one second</returns>
    public: std::uint64_t GetFrequency() const;

    /// <summary>Retrieves the time scaled to the performance counter frequency</summary>
    /// <returns>The time from timeGetTime() scaled to the performance counter</returns>
    /// <remarks>
    ///   This must not be called if the performance counter is not available.
    /// </remarks>
    private: std::uint64_t getTimeAsCounts() const;

    /// <summary>How often the performance counter ticks per second</summary>
    private: std::uint64_t performanceCounterFrequency;
    /// <summary>Previous value of the performance counter</summary>
    private: mutable std::uint64_t previousCounts;
    /// <summary>Offset of the performance counter relative to timeGetTime()</summary>
    private: mutable std::uint64_t performanceCounterOffset;
    /// <summary>The resolution of the millisecond fallback timer</summary>
    private: std::uint32_t timerResolution;
 
  };

  // ------------------------------------------------------------------------------------------- //

}}} // namespace Nuclex::Game::Timing

#endif // NUCLEX_GAME_TIMING_WINDOWSCLOCK_H
