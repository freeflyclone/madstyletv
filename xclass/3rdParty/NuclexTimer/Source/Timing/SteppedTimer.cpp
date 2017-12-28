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

#include "Timing/SteppedTimer.h"
#include "Timing/Clock.h"
#include <stdexcept>

namespace {

  // ------------------------------------------------------------------------------------------- //

  /// <summary>The number of microseconds in one second</summary>
  const std::uint64_t MicrosecondsPerSecond = 1000000;

  /// <summary>Step frequency the timer will used when initialized</summary>
  const std::size_t DefaultStepFrequency = 60;

  // ------------------------------------------------------------------------------------------- //

} // anonymous namespace

namespace Nuclex { namespace Game { namespace Timing {

  // ------------------------------------------------------------------------------------------- //

  SteppedTimer::SteppedTimer() : Timer(Clock::GetSystemDefault()),
    stepFrequency(DefaultStepFrequency),
    error(DefaultStepFrequency / 2),
    frameRealWorldUs(0),
    frameSimulationUs(0),
    totalRealWorldUs(0),
    totalSimulationUs(0) {}

  // ------------------------------------------------------------------------------------------- //

  SteppedTimer::SteppedTimer(const std::shared_ptr<Clock> &clock) : Timer(clock),
    stepFrequency(DefaultStepFrequency),
    error(DefaultStepFrequency / 2),
    frameRealWorldUs(0),
    frameSimulationUs(0),
    totalRealWorldUs(0),
    totalSimulationUs(0) {}

  // ------------------------------------------------------------------------------------------- //

  void SteppedTimer::Reset() {
    Timer::Reset();

    this->frameRealWorldUs = 0;
    this->frameSimulationUs = 0;
    this->totalRealWorldUs = 0;
    this->totalSimulationUs = 0;

    this->error = this->stepFrequency / 2;
  }

  // ------------------------------------------------------------------------------------------- //

  std::size_t SteppedTimer::GetStepFrequency() const {
    return this->stepFrequency;
  }

  // ------------------------------------------------------------------------------------------- //

  void SteppedTimer::SetStepFrequency(std::size_t stepFrequency) {
    if(stepFrequency != this->stepFrequency) {
      this->stepFrequency = stepFrequency;
      Reset();
    }
  }

  // ------------------------------------------------------------------------------------------- //

  bool SteppedTimer::TryAdvance(GameTime &gameTime) {
    // Calculate the number of microseconds required to complete the next time step
    uint64_t required = (MicrosecondsPerSecond - this->error) / this->stepFrequency;

    // In most cases (whenever the result isn't evenly dividable), we'll end up one
    // microsecond short of the required amount since integer divisions always round down.
    // If that's the case, compensate by requiring one microsecond more.
    uint64_t test = this->error + (required * this->stepFrequency);
    if(test < MicrosecondsPerSecond) {
      ++required;
    }

    // Now we can check whether enough real time has elapsed to step game time forward
    std::uint64_t elapsedUs = Timer::GetElapsedMicroseconds();
    if(elapsedUs >= required) {
      Timer::AddAccountedMicroseconds(required);

      // Accumulate the division error which takes care of left-over microseconds in
      // case the step frequency is not evenly dividable by microseconds.
      this->error += required * this->stepFrequency;
      this->error -= MicrosecondsPerSecond;

      this->totalRealWorldUs += required;
      this->frameRealWorldUs += required;

      if(IsSimulationPaused()) {
        gameTime.SimulationDeltaUs = 0;
      } else {
        gameTime.SimulationDeltaUs = required;
        this->totalSimulationUs += required;
        this->frameSimulationUs += required;
      }

      gameTime.RealWorldDeltaUs = required;
    } else {
      gameTime.SimulationDeltaUs = 0;
      gameTime.RealWorldDeltaUs = 0;
    }

    gameTime.RealWorldTotalUs = this->totalRealWorldUs;
    gameTime.SimulationTotalUs = this->totalSimulationUs;

    return (elapsedUs >= required);
  }
        
  // ------------------------------------------------------------------------------------------- //

  void SteppedTimer::ResetFrameTime() {
    this->frameRealWorldUs = 0;
    this->frameSimulationUs = 0;
  }

  // ------------------------------------------------------------------------------------------- //
    
  GameTime SteppedTimer::GetFrameTimeAndReset() {
    std::uint64_t frameRealWorldUs = this->frameRealWorldUs;
    std::uint64_t frameSimulationUs = this->frameSimulationUs;

    this->frameRealWorldUs = 0;
    this->frameSimulationUs = 0;

    return GameTime(
      this->totalSimulationUs,
      frameSimulationUs,
      this->totalRealWorldUs,
      frameRealWorldUs
    );
  }

  // ------------------------------------------------------------------------------------------- //
    
  GameTime SteppedTimer::GetFrameTime() const {
    return GameTime(
      this->totalSimulationUs,
      this->frameSimulationUs,
      this->totalRealWorldUs,
      this->frameRealWorldUs
    );
  }

  // ------------------------------------------------------------------------------------------- //

}}} // namespace Nuclex::Game::Timing
