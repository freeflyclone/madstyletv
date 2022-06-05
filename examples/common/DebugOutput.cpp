#include "DebugOutput.h"

void InitStdLog()
{
	static OutputDebugStringBuf charDebugOutput;
	std::cout.rdbuf(&charDebugOutput);
	std::clog.rdbuf(&charDebugOutput);
}
