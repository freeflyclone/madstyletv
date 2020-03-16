#pragma once

#include <stdlib.h>
#include <stdarg.h>

enum XLogLevel {
	XLLoud, XLTrace, XLDebug, XLWarning, XLError
};

#ifdef __cplusplus
#include "xobject.h"

#define XLOG_DECLARE(r,...) XLog::Logger logger(r,__VA_ARGS__)
#define XLOG(l,...) logger.Log((l), __FUNCTION__"  | " __VA_ARGS__)

namespace XLog {
	class Logger : public XObject
	{
	public:
		Logger(std::string);
		Logger(std::string, XLogLevel l = defaultLevel);
		virtual ~Logger();

		void Log(const char*fmt, va_list ap);
		void Log(const char *, ...);
		void Log(XLogLevel, const char*, ...);
	
	private:
		char buff[8192];
		static const XLogLevel defaultLevel{ XLError };
		XLogLevel currentLevel{ XLError };
	};
}
#endif //__cplusplus