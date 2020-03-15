#pragma once

#include <stdlib.h>
#include <stdarg.h>

extern "C" void DefaultLog(const char*, ...);

#ifdef __cplusplus
#include "xobject.h"

#define XLOG_DECLARE(x) XLog::Logger logger(x);
#define XLOG(s, ...) logger.Log(s, __VA_ARGS__)

namespace XLog {
	typedef enum {
		Error, Warning, Debug, Trace, Loud
	} Level;

	class Logger : public XObject
	{
	public:
		Logger(std::string);
		virtual ~Logger();

		void Log(const char *, ...);

	private:
		char buff[8192];
		Level defaultLevel{ Error };
		Level currentLevel{ Error };
	};
}
#endif //__cplusplus