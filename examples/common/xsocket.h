#ifndef XSOCKET_H
#define XSOCKET_H

#include <string>

#include "socket.h"

class XSocket {
public:
	XSocket();
	~XSocket();

	int Open(std::string addr, int port, int type = SOCK_STREAM, int proto = IPPROTO_TCP, int flag = false);
	int Close();
	int Connect();

	static std::string Host2Addr(std::string host);
	int GetLastError();

private:
	SOCKET m_socket{ INVALID_SOCKET };
	std::string m_addr;
	int m_port;
	int m_type{ SOCK_STREAM };
	int m_proto{ IPPROTO_TCP };
	int m_flag{ false };
	int m_error{ 0 };
};
#endif

