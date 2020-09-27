#ifndef XSOCKET_H
#define XSOCKET_H

#include <string>

#include "socket.h"

class XSocket {
public:
	XSocket();
	~XSocket();

	static std::string Host2Addr(std::string host);
};
#endif

