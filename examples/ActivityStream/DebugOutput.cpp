#include "DebugOutput.h"

void InitStdLog()
{
	static OutputDebugStringBuf charDebugOutput;
	std::clog.rdbuf(&charDebugOutput);
}
