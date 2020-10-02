#ifdef _WIN32
#include <windows.h>
#endif

#include <stdio.h>
#include <stdarg.h>
#include <string.h>

static bool isInitialized = false;
static CRITICAL_SECTION cs;

extern "C" int xprintf(char *fmt,...)
{
	static int _initialized = 0;
	if (!_initialized)
	{
		InitializeCriticalSection(&cs);
		_initialized = 1;
	}

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
