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

#include "Timing/ScaledTimer.h"
#include "Timing/Clock.h"

#include <stdexcept>

namespace Nuclex { namespace Game { namespace Timing {

  // ------------------------------------------------------------------------------------------- //

  ScaledTimer::ScaledTimer() : Timer(Clock::GetSystemDefault()),
    accumulatedSimulationUs(0),
    totalSimulationUs(0),
    totalRealWorldUs(0) {
    this->simulationResumeUs = Timer::GetElapsedMicroseconds();
  }

  // ------------------------------------------------------------------------------------------- //

  ScaledTimer::ScaledTimer(const std::shared_ptr<Clock> &clock) : Timer(clock),
    accumulatedSimulationUs(0),
    totalSimulationUs(0),
    totalRealWorldUs(0) {
    this->simulationResumeUs = Timer::GetElapsedMicroseconds();
  }

  // ------------------------------------------------------------------------------------------- //

  void ScaledTimer::PauseSimulation() {
    if(!IsSimulationPaused()) {
      std::uint64_t deltaUs = Timer::GetElapsedMicroseconds() - this->simulationResumeUs;
      this->accumulatedSimulationUs += deltaUs;

      Timer::PauseSimulation();
    }
  }

  // ------------------------------------------------------------------------------------------- //

  void ScaledTimer::ResumeSimulation() {
    if(IsSimulationPaused()) {
      this->simulationResumeUs = Timer::GetElapsedMicroseconds();

      Timer::ResumeSimulation();
    }
  }

  // ------------------------------------------------------------------------------------------- //
    
  void ScaledTimer::Reset() {
    Timer::Reset();

    this->simulationResumeUs = Timer::GetElapsedMicroseconds();
    this->accumulatedSimulationUs = 0;

    this->totalRealWorldUs = 0;
    this->totalSimulationUs = 0;
  }

  // ------------------------------------------------------------------------------------------- //

  GameTime ScaledTimer::GetElapsedAndResetDelta() {
    std::uint64_t elapsedUs = Timer::GetElapsedMicroseconds();
    Timer::AddAccountedMicroseconds(elapsedUs);

    std::uint64_t simulationDeltaUs = this->accumulatedSimulationUs;
    if(!IsSimulationPaused()) {
      simulationDeltaUs += elapsedUs - this->simulationResumeUs;
      this->simulationResumeUs = elapsedUs;
    }
    this->accumulatedSimulationUs = 0;

    this->totalRealWorldUs += elapsedUs;
    this->totalSimulationUs += simulationDeltaUs;

    return GameTime(
      this->totalSimulationUs,
      simulationDeltaUs,
      this->totalRealWorldUs,
      elapsedUs
    );
  }

  // ------------------------------------------------------------------------------------------- //

  GameTime ScaledTimer::GetElapsed() const {
    std::uint64_t elapsedUs = Timer::GetElapsedMicroseconds();

    std::uint64_t simulationDeltaUs = this->accumulatedSimulationUs;
    if(!IsSimulationPaused()) {
      simulationDeltaUs += elapsedUs - this->simulationResumeUs;
    }

    return GameTime(
      this->totalSimulationUs + simulationDeltaUs, // total simulation time
      simulationDeltaUs, // elapsed simulation time
      this->totalRealWorldUs + elapsedUs, // total real-world time
      elapsedUs // elapsed real-world time
    );
  }

  // ------------------------------------------------------------------------------------------- //

}}} // namespace Nuclex::Game::Timing
