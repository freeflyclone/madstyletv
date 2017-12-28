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

#ifndef NUCLEX_GAME_TIMING_SCALEDTIMER_H
#define NUCLEX_GAME_TIMING_SCALEDTIMER_H

#include "Timer.h"

namespace Nuclex { namespace Game { namespace Timing {

  // ------------------------------------------------------------------------------------------- //

  /// <summary>Timer designed to facilitate time-scaled updating</summary>
  /// <remarks>
  ///   <para>
  ///     Frame rate independent movement can be implemented in two ways: either via
  ///     time scaling or via time stepping. Time scaling will scale the movements of all
  ///     objects in a game by the amount of time passed since the last frame. Time stepping
  ///     advances time in fixed steps, multiple times if more time has passed than the length
  ///     of a single step.
  ///   </para>
  ///   <para>
  ///     This timer is intended for the scaled approach. Its advantage is a little less CPU
  ///     load on low end systems (where a stepped timer would be doing several steps each
  ///     cycle). However, physics simulations tend to get unstable if time jumps by larger
  ///     amounts and scaling object movements by time means that multiplayer states will
  ///     eventually drift apart due to different rounding errors occuring as a result of speed
  ///     differences between the connected systems.
  ///   </para>
  ///   <example>
  ///     <code>
  ///       scaledTimer.Reset();
  ///       while(!this->quitRequested) {
  ///         GameTime deltaTime = scaledTimer.GetElapsedAndReset();
  ///
  ///         UpdateAll(deltaTime);
  ///         DrawAll(deltaTime);
  ///
  ///         RunMessagePump();
  ///       }
  ///     </code>
  ///   </example>
  ///   <para>
  ///     The time returned from GetElapsed() or GetElapsedAndReset() is the number of
  ///     completely elapsed microseconds. The fractional part of the currently running
  ///     microsecond is kept until the next call to GetElapsed() or GetElapsedAndReset(),
  ///     guaranteeing that no time is ever lost. The same goes for when the provided Clock's
  ///     frequency is not evenly divisible to microseconds.
  ///   </para>
  /// </remarks>
  class ScaledTimer : public Timer {

    /// <summary>Initializes a new scaled timer using the default clock</summary>
    public: ScaledTimer();

    /// <summary>Initializes a new scaled timer using the specified clock</summary>
    /// <param name="clock">Clock the scaled timer will use</param>
    public: ScaledTimer(const std::shared_ptr<Clock> &clock);
      
    /// <summary>Destroys the scaled timer</summary>
    public: virtual ~ScaledTimer() {}

    /// <summary>Pauses the simulation clock</summary>
    /// <remarks>
    ///   Real world time will still continue running, but the timer's simulation time
    ///   will stop counting until the ResumeSimulation() or Reset() methods are called.
    /// </remarks>
    public: void PauseSimulation();

    /// <summary>Resumes the simulation clock after it has been paused</summary>
    public: void ResumeSimulation();

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
    public: void Reset();

    /// <summary>Returns the elapsed time since the timer was last reset</summary>
    /// <returns>The time that has elapsed since the last Reset() call</returns>
    /// <remarks>
    ///   This method is identical to GetElapsedAndReset() except that it doesn't reset
    ///   the delta times, allowing you to &quot;peek&quot; at what the timing values would
    ///   be right now. Do not use this method in your game loop to advance time due to
    ///   the effects described in the documentation of the Reset() method.
    /// </remarks>
    public: GameTime GetElapsed() const;

    /// <summary>
    ///   Returns the amount of time elapsed since the last call and resets the elapsed time
    /// </summary>
    /// <remarks>
    ///   <para>
    ///     Use this method in a time-scaled game loop to determine the amount of time that
    ///     has elapsed since the previous frame. Time will be advanced in such a way that
    ///     all of clocks ticks will be accounted for. If the elapsed time is not evenly
    ///     divisible to the next millisecond, the remaining clock ticks will be budgeted
    ///     for the next frame.
    ///   </para>
    /// </remarks>
    public: GameTime GetElapsedAndResetDelta();

    /// <summary>Accumulated simulation time during the intervals where it was running</summary>
    private: std::uint64_t accumulatedSimulationUs;
    /// <summary>Time since which the simulation clock has been running</summary>
    private: std::uint64_t simulationResumeUs;

    /// <summary>Total elapsed simulation time in microseconds</summary>
    private: std::uint64_t totalSimulationUs;
    /// <summary>Total elapsed time in the real world in microseconds</summary>
    private: std::uint64_t totalRealWorldUs;
 
  };

  // ------------------------------------------------------------------------------------------- //

}}} // namespace Nuclex::Game::Timing

#endif // NUCLEX_GAME_TIMING_WINDOWSTIMEPROVIDER_H
