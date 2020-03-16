#pragma once

#include <stdlib.h>
#include <stdarg.h>

enum XLogLevel {
	XLLoud, XLTrace, XLDebug, XLWarning, XLError
};

#ifdef __cplusplus
#include "xobject.h"

#define XLOG_DECLARE(x) XLog::Logger logger(x)
#define XLOG(f, ...) logger.Log(__FUNCTION__"  | "f, __VA_ARGS__)

namespace XLog {
	class Logger : public XObject
	{
	public:
		Logger(std::string);
		virtual ~Logger();

		void Log(const char *, ...);
	
	private:
		char buff[8192];
		XLogLevel defaultLevel{ XLError };
		XLogLevel currentLevel{ XLError };
	};
}
#endif //__cplusplus