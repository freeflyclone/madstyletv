#ifndef XSOCKET_H
#define XSOCKET_H

#include <string>

#include "socket.h"

class XSocket {
public:
	XSocket();
	~XSocket();

	SOCKET Open(std::string addr, int port, int type = SOCK_STREAM, int proto = IPPROTO_TCP, int flag = false);

	static std::string Host2Addr(std::string host);
private:
	SOCKET m_socket = -1;
};
#endif

