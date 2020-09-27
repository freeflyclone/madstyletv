#include "xsocket.h"

XSocket::XSocket() {
	SocketsSetup();
}

XSocket::~XSocket() {

}

std::string XSocket::Host2Addr(std::string host) {
	return SocketHostToAddr((char *)host.c_str());
}

SOCKET XSocket::Open(std::string addr, int port, int type, int proto, int flag) {
	m_socket = SocketOpen((char*)addr.c_str(), port, type, proto, flag);

	return m_socket;
}