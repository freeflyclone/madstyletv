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

#ifndef NUCLEX_GAME_TIMING_TIMER_H
#define NUCLEX_GAME_TIMING_TIMER_H

#include "GameTime.h"
#include <memory>
#include <cstdint>

namespace Nuclex { namespace Game { namespace Timing {

  // ------------------------------------------------------------------------------------------- //

  class Clock; // forward declaration so the header is not required

  // ------------------------------------------------------------------------------------------- //

  /// <summary>Timer for frame rate-independent updates</summary>
  /// <remarks>
  ///   <para>
  ///     This base class takes care of all the complicated details such as clock wraparounds,
  ///     clock frequencies that are not evenly dividable by microseconds and clocks that run
  ///     slower than one tick per microsecond.
  ///   </para>
  ///   <para>
  ///     All of the internal time keeping code is based on integers and takes great care not
  ///     to lose even a single clock tick. As a user of this class, all you need to know is
  ///     that as long as the clock runs at the speed it indicated, after one second exactly
  ///     1,000,000 microseconds (1 second) will have been counted by the timer with no lost
  ///     ticks or any inaccuracies.
  ///   </para>
  ///   <para>
  ///     <strong>Notes if you want to derive your own timer from this class:</strong>
  ///     Time is processed like a banking account: each passed microsecond is credited to
  ///     the timer (so GetElapsedMicroseconds() would check your current balance). Any
  ///     microseconds you provide to the game as elapsed time should be withdrawn from
  ///     the timer by calling AddAccountedMicroseconds() (this will reduce the number returned
  ///     from GetElapsedMicroseconds() by the withdrawn amount).
  ///   </para>
  /// </remarks>
  class Timer {

    /// <summary>Initializes a new timer</summary>
    /// <param name="clock">Clock the timer will be updated from</param>
    public: Timer(const std::shared_ptr<Clock> &clock);

    /// <summary>Destroys the timer</summary>
    public: virtual ~Timer() {}
        
    /// <summary>Whether the simulation clock is currently paused</summary>
    /// <returns>True if the simulation clock is currently paused, false otherwise</returns>
    public: virtual bool IsSimulationPaused() const {
      return this->isSimulationPaused;
    }

    /// <summary>Pauses the simulation clock</summary>
    /// <remarks>
    ///   Real world time will still continue running, but the timer's simulation time
    ///   will stop counting until the ResumeSimulation() or Reset() methods are called.
    /// </remarks>
    public: virtual void PauseSimulation() {
      this->isSimulationPaused = true;
    }

    /// <summary>Resumes the simulation clock after it has been paused</summary>
    public: virtual void ResumeSimulation() {
      this->isSimulationPaused = false;
    }
    
    /// <summary>Resets the delta times to zero</summary>
    /// <remarks>
    ///   <para>
    ///     You should call this method once immediately before entering the main loop so the
    ///     time accumulated between the time provider being created and your game becoming
    ///     ready to run will not result in a jump of possibly several seconds of game time
    ///     being skipped/caught up.
    ///   </para>
    ///   <para>
    ///     Do not use this method in your normal update loop as this will result in the time
    ///     that passes between retrieving the timer's delta time or steps and calling Reset(),
    ///     leading to minuscule speed differences depending on the operating system's CPU load
    ///     and performance.
    ///   </para>
    ///   <para>
    ///     If the simulation clock was paused, calling Reset() will also restore it running.
    ///   </para>
    /// </remarks>
    public: virtual void Reset();

    /// <summary>Calculates the number of microseconds that have elapsed</summary>
    /// <returns>The number of elapsed microseconds</returns>
    protected: std::uint64_t GetElapsedMicroseconds() const;

    /// <summary>Adds microseconds that the timer has accounted for</summary>
    /// <param name="microseconds">Number of microseconds the timer has accounted for</param>
    protected: void AddAccountedMicroseconds(std::uint64_t microseconds);

    /// <summary>
    ///   Takes all ticks from the timer that have accumulated since the last call
    /// </summary>
    /// <returns>The number of clock ticks accumulated since the last call</returns>
    /// <remarks>
    ///   This method is intended for internal usage only and handles timer wraparounds
    ///   as it advances the last known clock value. The caller has to take care of adding
    ///   the withdrawn ticks to some counter in order to present a consistent view to
    ///   any user of this class.
    /// </remarks>
    private: std::uint64_t withdrawAllClockTicks() const;

    /// <summary>Updates the scaling values used to adjust to the clock frequency</summary>
    private: void updateClockFrequency();

    private: Timer &operator =(const Timer &);
    private: Timer(const Timer &);

    /// <summary>Whether simulation time is currently paused</summary>
    private: bool isSimulationPaused;

    /// <summary>Clock the timer is using to measure passing time</summary>
    private: const std::shared_ptr<Clock> clock;
    /// <summary>Frequency at which the clock is running</summary>
    private: std::uint64_t clockFrequency;
    /// <summary>Maximum value the clock can assume before wrapping around</summary>
    private: std::uint64_t wraparoundTicks;

    /// <summary>Clock ticks the timer has already accounted for</summary>
    private: mutable std::uint64_t lastClockTicks;
    /// <summary>Error accumulator in units of MHz x clock frequency</summary>
    private: mutable std::uint64_t error;
    /// <summary>Microseconds the AddAccountedMicroseconds() method could not process</summary>
    private: mutable std::uint64_t unaccountedMicroseconds;
        
  };

  // ------------------------------------------------------------------------------------------- //

}}} // namespace Nuclex::Game::Timing

#endif // NUCLEX_GAME_TIMING_TIMER_H
