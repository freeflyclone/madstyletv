/**************************************************************
** main.cpp
**
** TimingTest project. Experiments with Nuclex.Timer, hoping
** to find it suitable for streaming playback;
**
** Specifically: most streaming container formats specify
** various synchonization clocks: current stream time, 
** access unit presentation times, and for video: decode
** times vs. presentation times to allow for frame reordering
** due to bi-directional motion estimation.
**
** Smooth playback involves synchronization presentation of
** audio and video units.  It is confounded on desktop OS's
** by lack of precision in thread scheduling.
*************************************************************/
#include <iostream>

#include "Timing/SteppedTimer.h"
#include "Timing/ScaledTimer.h"

using namespace std;
using namespace Nuclex::Game::Timing;

void moveCursorOneLineUp();

int main() {
	SteppedTimer steppedTimer;
	ScaledTimer scaledTimer;

	// How many steps per second the stepped timer should perform. If this
	// number is not an even divisor of one second, the time steps will
	// vary in size so that after 1 second, exactly 1 second of delta time
	// will have accumulated.
	steppedTimer.SetStepFrequency(60); // 6 Hz

	std::size_t stepIndex = 0;
	for (;;) {
		GameTime gameTime;

		// The stepped timer will only advance time when more time has accumulated
		// than a step is long. Try holding the program with Ctrl+S, wait a second
		// and continue with Ctrl+Q to see how it catches up again if enough time
		// for multiple steps has elapsed!
		while (steppedTimer.TryAdvance(gameTime)) {
			cout << "Time stepped " << gameTime.RealWorldDeltaUs << " microseconds (step #" << stepIndex++ << ")" << endl;
		}

		// The scaled timer simply continues running and returns the exact time
		// in microsecond whenever it is asked.
		gameTime = scaledTimer.GetElapsedAndResetDelta();
		cout << "Running for " << gameTime.RealWorldTotalUs	<< " microseconds" << endl;

		moveCursorOneLineUp();
	}

	return 0;
}

// ------------------------------------------------------------------------- //

// Some win32 stuff to beautify output. If your system is not Windows but
// has gotoxy(), feel free to add your own implementation if this function :)

#if defined(WIN32)
#include <windows.h>
#endif

void moveCursorOneLineUp() {
#if defined(WIN32)
	HANDLE consoleHandle = ::GetStdHandle(STD_OUTPUT_HANDLE);

	CONSOLE_SCREEN_BUFFER_INFO bufferInfo = { 0 };
	::GetConsoleScreenBufferInfo(consoleHandle, &bufferInfo);

	COORD cursorPosition = { 0 };
	cursorPosition.X = 0;
	cursorPosition.Y = bufferInfo.dwCursorPosition.Y - 1;
	::SetConsoleCursorPosition(consoleHandle, cursorPosition);
#endif
}
