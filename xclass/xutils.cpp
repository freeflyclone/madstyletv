#ifdef _WIN32
#include <windows.h>
#endif

#include <stdio.h>
#include <stdarg.h>
#include <string.h>

namespace {
#ifdef _WIN32
	CRITICAL_SECTION cs;

	class CSInitializer {
	public:
		CSInitializer() { InitializeCriticalSection(&cs); }
	};

	CSInitializer csi;
#endif
}


extern "C" int xprintf(char *fmt,...)
{
	static char buff[8192];
	va_list ap;

#ifdef _WIN32
	EnterCriticalSection(&cs);
#endif

	va_start(ap, fmt);

#ifdef _WIN32
    vsprintf_s(buff, sizeof(buff), fmt, ap);
    //strncat_s(buff, sizeof(buff), "\n", 1);
    OutputDebugStringA(buff);

	LeaveCriticalSection(&cs);
#else
    vsprintf(buff,fmt,ap);
    printf("%s", buff);
#endif

	return 0;
}

#ifdef _MACOSX
#error fix this
#endif
