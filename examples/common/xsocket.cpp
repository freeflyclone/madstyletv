#include "xsocket.h"

// Static object initialization of file-scope object "_xsock"
// forces XSocket constructor to call SocketsSetup at LOAD time which, on 
// Windows, causes the Winsock initializer WSAStartup() to be called.
//
// From then on, Berkeley sockets semantics are mostly like POSIX counterparts.
const static XSocket _xsock;

XSocket::XSocket() {
	SocketsSetup();
}

XSocket::~XSocket() {
	Close();
}

std::string XSocket::Host2Addr(std::string host) {
	const char* addrStr = SocketHostToAddr((char*)host.c_str());
	
	if (addrStr)
		return addrStr;

	return "NotFound";
}

int XSocket::GetLastError() {
	int lastError = m_error;
	m_error = 0;
	return lastError;
}

int XSocket::Open(std::string addr, int port, int type, int proto, int flag) {
	m_addr = addr;
	m_port = port;
	m_type = type;
	m_proto = proto;
	m_flag = flag;

	m_socket = SocketOpen((char*)m_addr.c_str(), m_port, m_type, m_proto, m_flag);
	if (m_socket == INVALID_SOCKET)
		m_error = WSAGetLastError();

	return (int)m_socket;
}

int XSocket::Close() {
	int retVal = SocketClose(m_socket);
	if (retVal == SOCKET_ERROR)
		m_error = WSAGetLastError();

	return retVal;
}

int XSocket::Connect() {
	sockaddr_in sai;
	int retVal;

	SockAddrIN(&sai, (char*)m_addr.c_str(), m_port);

	retVal = connect(m_socket, (SOCKADDR*)&sai, sizeof(sai));
	if (retVal == SOCKET_ERROR)
		m_error = WSAGetLastError();

	return retVal;
}

int XSocket::Send(const char *buffer, int length) {
	sockaddr_in sai;
	int retVal = 0;

	SockAddrIN(&sai, (char*)m_addr.c_str(), m_port);

	retVal = sendto(m_socket, buffer, length, 0, (SOCKADDR *)&sai, sizeof(sai));
	if (retVal == SOCKET_ERROR)
		m_error = WSAGetLastError();

	return retVal;
}

int XSocket::Recv(char *buffer, int size) {
	sockaddr_in sai;
	int nRead = 0;

	SockAddrIN(&sai, (char*)m_addr.c_str(), m_port);

	nRead = recv(m_socket, buffer, size, 0);
	if (nRead == SOCKET_ERROR)
		m_error = WSAGetLastError();

	return nRead;


}