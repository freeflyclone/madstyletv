#include "xlog.h"

#include <stdlib.h>
#include <stdarg.h>

#ifdef _WIN32
#include <windows.h>
#endif

using namespace XLog;

Logger::Logger(std::string n) : XObject(n)
{
	sprintf(buff, "%20s|%s|\n", n.c_str(), __FUNCTION__);
	OutputDebugStringA(buff);
}

Logger::Logger(std::string n, XLogLevel l) : XObject(n), currentLevel(l)
{
	sprintf(buff, "%20s|%s|level: %d\n", n.c_str(), __FUNCTION__, l);
	OutputDebugStringA(buff);
}

Logger::~Logger()
{
	sprintf(buff, "%20s|%s|\n", Name().c_str(), __FUNCTION__);
	OutputDebugStringA(buff);
}

void Logger::Log(const char*fmt, va_list ap)
{
#ifdef _WIN32
	sprintf(buff, "%20s|", Name().c_str());
	vsprintf_s(buff + strlen(buff), sizeof(buff), fmt, ap);
	OutputDebugStringA(buff);
	OutputDebugStringA("\n");
#else
	vsprintf(buff, fmt, ap);
	printf("%s", buff);
#endif
}

void Logger::Log(const char*fmt, ...) {
	va_list ap;
	va_start(ap, fmt);
	Log(fmt, ap);
}

void Logger::Log(XLogLevel level, const char* fmt, ...)
{
	if (level < currentLevel)
		return;

	va_list ap;
	va_start(ap, fmt);
	Log(fmt, ap);
}