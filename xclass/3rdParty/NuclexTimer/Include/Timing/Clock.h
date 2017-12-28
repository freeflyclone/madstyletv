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

#ifndef NUCLEX_GAME_TIMING_CLOCK_H
#define NUCLEX_GAME_TIMING_CLOCK_H

#include <cstdint>
#include <memory>

namespace Nuclex { namespace Game { namespace Timing {

  // ------------------------------------------------------------------------------------------- //

  /// <summary>Provides accurate timings at system-defined resolutions</summary>
  /// <remarks>
  ///   Clocks must be steady, meaning that they never jump backwards due to daylight saving
  ///   time, NTP synchronization or inaccuracies. If a time source is prone to jumping
  ///   backwards, you should ensure its integrity by cross-checking it with another time
  ///   source and bridge any values that seem implausible with the alternate time source.
  /// </remarks>
  class Clock {

    /// <summary>Destroys the clock</summary>
    public: virtual ~Clock() {}

    /// <summary>Returns the default clock for the current system</summary>
    public: static std::shared_ptr<Clock> GetSystemDefault();

    /// <summary>
    ///   Retrieves the maximum value the time can assume before wrapping around to 0 again
    /// </summary>
    /// <returns>The highest possible value GetTime() can return</returns>
    public: virtual std::uint64_t GetWraparoundTime() const = 0;

    /// <summary>Retrieves the clock's current time</summary>
    /// <returns>The number of ticks the clock has ticked to far</returns>
    /// <remarks>
    ///   There's no rule for what a clock's time should be relative to, only that it
    ///   keeps counting at a fixed pace. The clock could begin ticking when the system
    ///   boots, when the application starts or when the class is instantiated.
    /// </remarks>
    public: virtual std::uint64_t GetTime() const = 0;

    /// <summary>Retrieves the clock's tick frequency</summary>
    /// <returns>How often the clock will tick in one second</returns>
    public: virtual std::uint64_t GetFrequency() const = 0;
 
  };

  // ------------------------------------------------------------------------------------------- //

}}} // namespace Nuclex::Game::Timing

#endif // NUCLEX_GAME_TIMING_CLOCK_H
