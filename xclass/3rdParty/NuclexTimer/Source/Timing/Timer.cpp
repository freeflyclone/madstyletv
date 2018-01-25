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

#include "Timing/Timer.h"
#include "Timing/Clock.h"

#include <cassert>

namespace {

  /// <summary>The number of microseconds in one second</summary>
  const std::uint64_t MicrosecondsPerSecond = 1000000;

} // anonymous namespace

namespace Nuclex { namespace Game { namespace Timing {

  // ------------------------------------------------------------------------------------------- //

  Timer::Timer(const std::shared_ptr<class Clock> &clock) :
    clock(clock),
    isSimulationPaused(false),
    lastClockTicks(clock->GetTime()),
    wraparoundTicks(clock->GetWraparoundTime()),
    unaccountedMicroseconds(0) {
    updateClockFrequency();
    this->error = this->clockFrequency / 2;
  }

  // ------------------------------------------------------------------------------------------- //

  void Timer::Reset() {
    this->isSimulationPaused = false;
    this->lastClockTicks = clock->GetTime();
    this->error = this->clockFrequency / 2;
    this->unaccountedMicroseconds = 0;
  }

  // ------------------------------------------------------------------------------------------- //

  std::uint64_t Timer::GetElapsedMicroseconds() const {
    std::uint64_t elapsedTicks = withdrawAllClockTicks();

    // For each whole second, we can just subtract the clock frequency.
    // After this, elapsed ticks are guaranteed to be less than the clock frequency.
    std::uint64_t seconds = elapsedTicks / this->clockFrequency;
    elapsedTicks -= seconds * this->clockFrequency;

    // Apply Bresenham's optimized integer algorithm: for each elapsed clock tick,
    // sum up the number of microseconds. Each time the sum passes the clock frequency,
    // subtract the flock frequency from the sum and count up by one microsecond.
    // This is more accurate than any floating point could ever be.
    this->error += (elapsedTicks * MicrosecondsPerSecond);
    std::uint64_t microseconds = this->error / this->clockFrequency;
    this->error -= microseconds * this->clockFrequency;

    // Add the newly elapsed time to the as of yet unaccounted microseconds to
    // present a consistent view to the user (we're a const method after all)
    this->unaccountedMicroseconds += seconds * MicrosecondsPerSecond + microseconds;

    return this->unaccountedMicroseconds;
  }

  // ------------------------------------------------------------------------------------------- //

  void Timer::AddAccountedMicroseconds(std::uint64_t microseconds) {
    using namespace std;
    assert(
      (microseconds <= this->unaccountedMicroseconds) &&
      "Tried to account for more microseconds than have elapsed"
    );

    this->unaccountedMicroseconds -= microseconds;
  }

  // ------------------------------------------------------------------------------------------- //

  void Timer::updateClockFrequency() {
    this->clockFrequency = this->clock->GetFrequency();
  }

  // ------------------------------------------------------------------------------------------- //

  std::uint64_t Timer::withdrawAllClockTicks() const {
    std::uint64_t ticks;

    // Assumption: accountedTicks is always a short time behind the clock, so we do
    // not have to handle the case where accountedTicks has wrapped around before
    // the clock wraps around. If the clock can jump back, prepare for a world of hurt.
    std::uint64_t time = this->clock->GetTime();
    if(time < this->lastClockTicks) { // did the clock wrap around?
      ticks = this->wraparoundTicks - this->lastClockTicks + 1 + time;
    } else {
      ticks = time - this->lastClockTicks;
    }

    // Update the previous clock ticks so 
    this->lastClockTicks = time;

    return ticks;
  }

  // ------------------------------------------------------------------------------------------- //

}}} // namespace Nuclex::Game::Timing
