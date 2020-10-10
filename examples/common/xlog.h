#pragma once

#include <stdlib.h>
#include <stdarg.h>


#define FUNCIN(...) xprintf("%s(%d) -->\n", __FUNCTION__, std::this_thread::get_id())
#define FUNCOUT(...) xprintf("%s(%d) <--\n", __FUNCTION__, std::this_thread::get_id())
#define FUNC(...) { xprintf("%s(%d): ", __FUNCTION__, std::this_thread::get_id()) ; xprintf(__VA_ARGS__); }
#define FLESS(...) { xprintf(" %s:%d (%d) : ", __FILE__, __LINE__, std::this_thread::get_id()) ; xprintf(__VA_ARGS__); }

enum XLogLevel {
	XLLoud, XLTrace, XLDebug, XLWarning, XLError
};

#ifdef __cplusplus
#include "xobject.h"
#include <mutex>

#define XLOG_DEFINE(r,...) XLog::Logger logger(r,##__VA_ARGS__)
#define XLOG(l,...) { logger.Log((l), __FUNCTION__, ##__VA_ARGS__); }

namespace XLog {
	class Logger : public XObject
	{
	public:
		Logger::Logger(std::string n) : XObject(n) {}
		Logger::Logger(std::string n, XLogLevel l) : XObject(n), currentLevel(l) {}
		virtual Logger::~Logger() {}

		void Log(const char*fmt, va_list ap);
		void Log(const char *, ...);
		void Log(XLogLevel, const char*, ...);
		void Log(XLogLevel, const char*, const char*, ...);
	
	private:
		char buff[8192];
		static const XLogLevel defaultLevel{ XLError };
		XLogLevel currentLevel{ XLError };
		std::mutex mutexLock;

		char timestamp[128];
		const char* MakeTimestamp();
		unsigned long epoch;
	};
}
#endif //__cplusplus