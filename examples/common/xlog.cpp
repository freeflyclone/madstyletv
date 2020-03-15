#include "xlog.h"

#include <stdlib.h>
#include <stdarg.h>

#ifdef _WIN32
#include <windows.h>
#endif

using namespace XLog;

Logger::Logger(std::string n) : XObject(n)
{
	char buff[] = __FUNCTION__ "()\n";
	OutputDebugStringA(buff);
}

Logger::~Logger()
{
	char buff[] = __FUNCTION__ "()\n";
	OutputDebugStringA(buff);
}

void Logger::Log(const char*fmt, ...) {
	va_list ap;

	va_start(ap, fmt);

#ifdef _WIN32
	sprintf(buff, "%20s: ", Name().c_str());
	vsprintf_s(buff + strlen(buff), sizeof(buff), fmt, ap);
	OutputDebugStringA(buff);
	OutputDebugStringA("\n");
#else
	vsprintf(buff, fmt, ap);
	printf("%s", buff);
#endif
}
