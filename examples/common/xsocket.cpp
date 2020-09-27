#include "xsocket.h"

XSocket::XSocket() {
	SocketsSetup();
}

XSocket::~XSocket() {
	Close();
}

std::string XSocket::Host2Addr(std::string host) {
	return SocketHostToAddr((char *)host.c_str());
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

	sai.sin_family = AF_INET;
	sai.sin_addr.s_addr = inet_addr(m_addr.c_str());
	sai.sin_port = htons(m_port);

	retVal = connect(m_socket, (SOCKADDR*)&sai, sizeof(sai));
	if (retVal == SOCKET_ERROR)
		m_error = WSAGetLastError();

	return retVal;
}