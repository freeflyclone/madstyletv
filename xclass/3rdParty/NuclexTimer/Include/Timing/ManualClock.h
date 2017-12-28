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

#ifndef NUCLEX_GAME_TIMING_MANUALCLOCK_H
#define NUCLEX_GAME_TIMING_MANUALCLOCK_H

#include "Clock.h"

#include <cstddef>

namespace Nuclex { namespace Game { namespace Timing {

  // ------------------------------------------------------------------------------------------- //

  /// <summary>A clock that is advanced by hand</summary>
  class ManualClock : public Clock {

    /// <summary>Initializes a new manually advanced clock</summary>
    /// <param name="frequency">Frequency at which the clock ticks</param>
    public: ManualClock(std::uint64_t frequency = 1000) :
      elapsedTicks(0),
      frequency(frequency) {}

    /// <summary>Destroys the clock</summary>
    public: virtual ~ManualClock() {}

    /// <summary>Advances the clock's time by the specified number of ticks</summary>
    /// <param name="ticks">Number of ticks the clock will be advanced by</param>
    public: void AddTime(std::uint64_t ticks);

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

    /// <summary>Number of ticks that have elapsed on the clock</summary>
    private: std::uint64_t elapsedTicks;
    /// <summary>Frequency at which the clock ticks</summary>
    private: std::uint64_t frequency;
 
  };

  // ------------------------------------------------------------------------------------------- //

}}} // namespace Nuclex::Game::Timing

#endif // NUCLEX_GAME_TIMING_MANUALCLOCK_H
