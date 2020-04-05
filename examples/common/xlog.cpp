#include <stdlib.h>
#include <stdarg.h>
#ifdef _WIN32
#include <windows.h>
#endif
#include "xlog.h"

using namespace XLog;

const char* Logger::MakeTimestamp() {
	return "right-fucking-now";
}

void Logger::Log(const char*fmt, ...)
{
	va_list ap;
	va_start(ap, fmt);
	Log(fmt, ap);
}

void Logger::Log(XLogLevel level, const char* fmt, ...)
{
	if (level < currentLevel)
		return;

	int newSize = strlen(fmt) + 64;
	char* newFmt = new char[newSize];
	snprintf(newFmt, newSize, "%26s|", fmt);

	va_list ap;
	va_start(ap, fmt);
	Log(newFmt, ap);
}

void Logger::Log(XLogLevel level, const char* func, const char* fmt, ...)
{
	if (level >= currentLevel)
		return;

	int newSize = strlen(fmt) + 64;
	char* newFmt = new char[newSize];
	snprintf(newFmt, newSize, "%26s| %s", func, fmt);

	va_list ap;
	va_start(ap, fmt);
	Log(newFmt, ap);
}

void Logger::Log(const char*fmt, va_list ap)
{
	std::lock_guard<std::mutex> lock(mutexLock);

#ifdef _WIN32
	sprintf(buff, "%20s|", Name().c_str());
	int prefixLength = strlen(buff);
	vsprintf_s(buff + prefixLength, sizeof(buff)-prefixLength, fmt, ap);
	strcat(buff, "\n");
	OutputDebugStringA(buff);
#else
	vsprintf(buff, fmt, ap);
	printf("%s", buff);
#endif
}
