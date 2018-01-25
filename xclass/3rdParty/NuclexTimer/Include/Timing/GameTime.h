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

#ifndef NUCLEX_GAME_TIMING_GAMETIME_H
#define NUCLEX_GAME_TIMING_GAMETIME_H

#include <cstdint>

namespace Nuclex { namespace Game { namespace Timing {

  // ------------------------------------------------------------------------------------------- //

  /// <summary>Provides the timing values for the game's state update</summary>
  class GameTime {

    /// <summary>Initializes a new game timing carrier with zero values</summary>
    public: GameTime() :
      SimulationTotalUs(0),
      SimulationDeltaUs(0),
      RealWorldTotalUs(0),
      RealWorldDeltaUs(0) {}
      

    /// <summary>Initializes a new game timing carrier</summary>
    /// <param name="simulationTotalUs">Total microseconds that have passed in-game</param>
    /// <param name="simulationDeltaUs">Microseconds since the last update in-game</param>
    /// <param name="realWorldTotalUs">
    ///   Total microseconds that have passed in the real world
    /// </param>
    /// <param name="realWorldDeltaUs">
    ///   Microseconds since the last update in the real world
    /// </param>
    public: GameTime(
      std::uint64_t simulationTotalUs,
      std::uint64_t simulationDeltaUs,
      std::uint64_t realWorldTotalUs,
      std::uint64_t realWorldDeltaUs
    ) :
      SimulationTotalUs(simulationTotalUs),
      SimulationDeltaUs(simulationDeltaUs),
      RealWorldTotalUs(realWorldTotalUs),
      RealWorldDeltaUs(realWorldDeltaUs) {}

    /// <summary>Total number of microseconds that have passed in-game</summary>
    /// <remarks>
    ///   <para>
    ///     This clock will only continue to count when the game is running and not paused.
    ///     Use it for things dependent on the passing of time within the game, such as
    ///     movement, physics, rain and water animation. Do not use it for animations that
    ///     should keep playing even when the game is paused like animated menus or
    ///     loading screens.
    ///   </para>
    /// </remarks>
    public: std::uint64_t SimulationTotalUs;

    /// <summary>Microseconds that have passed since that last update in-game</summary>
    /// <remarks>
    ///   <para>
    ///     This clock will only run when the game is running and not paused. Use it for
    ///     things that depend on the passing of time within the game, such as movement,
    ///     physics, rain and water animation. Do not use it for animations that should
    ///     keep playing even when the game is paused like animated menus or loading screens.
    ///   </para>
    /// </remarks>
    public: std::uint64_t SimulationDeltaUs;

    /// <summary>Total number of real world microseconds the game has been running</summary>
    /// <remarks>
    ///   This clock will continue to run even when the game is paused. It should be used
    ///   for the timing of elements outside your game's world such as your menu system,
    ///   fps timing and so on. Do not use it for in-game events and game object movement.
    /// </remarks>
    public: std::uint64_t RealWorldTotalUs;

    /// <summary>Real world microseconds that have elapsed since the last update</summary>
    /// <remarks>
    ///   This clock will continue to run even when the game is paused. It should be used
    ///   for the timing of animated menus or loading animations where the in-game time might
    ///   not be advancing. Do not use it for in-game events and game object movement.
    /// </remarks>
    public: std::uint64_t RealWorldDeltaUs;

  };

  // ------------------------------------------------------------------------------------------- //

}}} // namespace Nuclex::Game::Timeing

#endif // NUCLEX_GAME_TIMING_GAMETIME_H
