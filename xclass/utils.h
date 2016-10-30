#ifndef DEBUG_H
#define DEBUG_H

#define if_debug(expression)	if (expression)
#define DebugFlush()			

#ifdef __cplusplus
extern "C" 
{
#endif

int DebugPrintfRegisterCallback(void (*Callback)(char *));
int DebugPrintfUnRegisterCallback(void);
void DebugPrintf(char *fmt,...);
void MBPrintf(char *fmt,...);
void ErrorPrintf(char *fmt,...);
void ExPrintf(char *fmt,...);
void error(char *fmt,...);
void PrintError(void);
void Sprint(int, int);
BOOL ByteSwap(PVOID, PVOID, DWORD);

#ifdef __cplusplus
};
#endif

#endif
