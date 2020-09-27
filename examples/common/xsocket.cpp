#include "xsocket.h"

XSocket::XSocket() {
	SocketsSetup();
}

XSocket::~XSocket() {

}

std::string XSocket::Host2Addr(std::string host) {
	return SocketHostToAddr((char *)host.c_str());
}