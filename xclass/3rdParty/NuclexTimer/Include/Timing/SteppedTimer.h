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

#ifndef NUCLEX_GAME_TIMING_STEPPEDTIMER_H
#define NUCLEX_GAME_TIMING_STEPPEDTIMER_H

#include "Timer.h"
#include <memory>
#include <cstdint>

namespace Nuclex { namespace Game { namespace Timing {

  // ------------------------------------------------------------------------------------------- //
    
  class Clock; // forward declaration so the header isn't required

  // ------------------------------------------------------------------------------------------- //

  /// <summary>Timer for games that advances time in fixed steps</summary>
  /// <remarks>
  ///   <para>
  ///     Frame rate independent movement can be implemented in two ways: either via
  ///     time scaling or via time stepping. Time scaling will scale the movements of all
  ///     objects in a game by the amount of time passed since the last frame. Time stepping
  ///     advances time in fixed steps, multiple times if more time has passed than the length
  ///     of a single step.
  ///   </para>
  ///   <para>
  ///     This timer is intended for the stepped approach. The advantages are simpler updating,
  ///     improved stability of physics simulations (most physics engines go haywire if time
  ///     jumps by a large amount) and the guarantee for identical rounding errors for all
  ///     players in a multi-player game.
  ///   </para>
  ///   <example>
  ///     <code>
  ///       steppedTimer.Reset();
  ///       while(!this->quitRequested) {
  ///         GameTime timeStep;
  ///         while(steppedTimer.TryAdvance(timeStep)) {
  ///           UpdateAll(timeStep);
  ///         }
  ///
  ///         GameTime frameTime = steppedTimer.GetFrameTimeAndReset()
  ///         DrawAll(frameTime);
  ///
  ///         RunMessagePump();
  ///       }
  ///     </code>
  ///   </example>
  ///   <para>
  ///     Delta times might not always be the same size. For example, if you decided to run
  ///     your game at 60 Hz, you would observe a repeating pattern of this:
  ///   </para>
  ///   <pre>
  ///     16667 microseconds
  ///     16666 microseconds
  ///     16667 microseconds
  ///     16667 microseconds
  ///     16666 microseconds
  ///     16667 microseconds
  ///     16667 microseconds
  ///     16666 microseconds
  ///     16667 microseconds
  ///   </pre>
  ///   <para>
  ///     Note the even distribution of some 16,666 microsecond steps, guaranteeing that after
  ///     one second, your game will have advanced by 1,000,000 microseconds and not
  ///     999,960 (16,666 x 60) or 1,000,020 (16,667 x 60) as would have resulted from simply
  ///     calculating the step size in microseconds.
  ///   </para>
  ///   <para>
  ///     If you want a fixed, unvarying floating point delta by which your game's time is
  ///     advanced, you can do so, too: since you know that 60 steps will be generated per
  ///     second without fail, you can simply affix your delta time to (1.0 / 60.0) and ignore
  ///     the deltas provided to you by the stepped timer.
  ///   </para>
  /// </remarks>
  class SteppedTimer : public Timer {

    /// <summary>Initializes a new stepped timer using the default system clock</summary>
    public: SteppedTimer();

    /// <summary>Initializes a new stepped timer using the specified clock</summary>
    /// <param name="clock">Clock the stepped timer will use</param>
    public: SteppedTimer(const std::shared_ptr<Clock> &clock);
      
    /// <summary>Destroys the stepped timer</summary>
    public: virtual ~SteppedTimer() {}
    
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

    /// <summary>Retrieves the number of steps per second the timer will produce</summary>
    /// <returns>The number of steps time that will be generated per second</returns>
    public: std::size_t GetStepFrequency() const;

    /// <summary>Sets the number of steps per second the timer will produce</summary>
    /// <param name="stepFrequency">New number of steps time is advanced per second</param>
    /// <remarks>
    ///   Altering the step frequency will reset the timer.
    /// </remarks>
    public: void SetStepFrequency(std::size_t stepFrequency);

    /// <summary>Attempts to advance the time by one step</summary>
    /// <param name="gameTime">
    ///   Receives the timings to which time has been advanced. If not enough time has passed,
    ///   will receive the timings of the current step.
    /// </param>
    /// <returns>
    ///   True if enough time has accumulated and time was stepped forward
    /// </returns>
    /// <remarks>
    ///   <para>
    ///     Call this method repeatedly until it returns false during your game's update
    ///     cycle, then advance to the drawing cycle where you can retrieve the total time
    ///     you have stepped forward via the GetFrameTime() method (don't forget to call
    ///     ResetFrameTime() after that).
    ///   </para>
    /// </remarks>
    public: bool TryAdvance(GameTime &gameTime);
        
    /// <summary>Resets the accumulated frame time</summary>
    /// <remarks>
    ///   <para>
    ///     Whenever time is successfully stepped forward via TryAdvance(), the amount of time
    ///     stepped forward is accumulated so it can be queried for the draw cycle via
    ///     GetFrameTime() and then reset through this method.
    ///   </para>
    /// </remarks>
    public: void ResetFrameTime();
    
    /// <summary>Determines the frame time that has accumulated since the last reset</summary>
    /// <returns>The amount of time that has been stepped forward since the last reset</returns>
    /// <remarks>
    ///   <para>
    ///     Whenever time is successfully stepped forward via TryAdvance(), the amount of time
    ///     stepped forward is accumulated so it can be queried for the draw cycle via this
    ///     method and then reset through ResetFrameTime().
    ///   </para>
    /// </remarks>
    public: GameTime GetFrameTime() const;

    /// <summary>Returns the accumulated frame time and resets it</summary>
    /// <returns>The frame time accumulated from all steps since the last reset</returns>
    /// <remarks>
    ///   This is just a convenience method that can be used during the game's draw cycle
    ///   to shorten the code needed to obtain the time steps time was advanced by since
    ///   the last frame and reset the frame time in one call.
    /// </remarks>
    public: GameTime GetFrameTimeAndReset();

    /// <summary>Steps the timer will generate per second</summary>
    private: std::size_t stepFrequency;
    /// <summary>Error accumulator in units of MHz x step frequency</summary>
    private: std::uint64_t error;

    /// <summary>Accumulated advanced real world time for the current frame</summary>
    private: std::uint64_t frameRealWorldUs;
    /// <summary>Accumulated advanced simulation time for the current frame</summary>
    private: std::uint64_t frameSimulationUs;
    /// <summary>Microseconds that have passed in the real world</summary>
    private: std::uint64_t totalRealWorldUs;
    /// <summary>Microseconds that have passed in simulation time</summary>
    private: std::uint64_t totalSimulationUs;

  };

  // ------------------------------------------------------------------------------------------- //

}}} // namespace Nuclex::Game::Timing

#endif // NUCLEX_GAME_TIMING_STEPPEDTIMER_H
