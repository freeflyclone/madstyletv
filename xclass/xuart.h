#ifndef XUART_H
#define XUART_H

#include <string>
#include <sstream>
#include "xutils.h"

#ifdef WIN32
#include <Windows.h>
class XUart {
public:
	XUart(std::string portName);

	~XUart();
	int Read(unsigned char *b, int size);
	int Write(unsigned char *b, int size);

private:
	HANDLE hPort;
	DCB portDCB;
	COMMTIMEOUTS   commTimeouts;
};
#else

#include <stdio.h>
#include <termios.h>
#include <unistd.h>
#include <fcntl.h>
#include <string.h>

class XUart {
public: 
	XUart(std::string);
	~XUart();

	int Read(unsigned char *b, int size);
	int Write(unsigned char *b, int size);

private:
	int portFd;
};
#endif

#endif // XUART_H
