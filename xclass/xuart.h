#ifndef XUART_H
#define XUART_H

#include <string>
#include <sstream>
#include <Windows.h>
#include "xutils.h"

class XUart {
public:
	XUart(std::string portName);

	~XUart();
	int Read(unsigned char *b, int size);

private:
	HANDLE hPort;
	DCB portDCB;
	COMMTIMEOUTS   commTimeouts;
	unsigned char buffer[1024];
};

#endif