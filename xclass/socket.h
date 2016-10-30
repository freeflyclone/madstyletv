// from MobiusDigitalVideo source code
#ifndef SOCKET_H
#define SOCKET_H

#define IN_CLASSD(i)	(((long)(i) & 0xf0000000) == 0xe0000000)
#define IN_MULTICAST(i)	IN_CLASSD(i)

#define socket_error printf

#ifdef __cplusplus
extern "C" {
#endif

#ifdef _WIN32
#include <WinSock2.h>
#include <ws2ipdef.h>
#endif

int SocketsSetup();
char *SocketHostAddr();
char *SocketHostName();
char *SocketHostToAddr(char *host);
SOCKET SocketOpen(char *addr, int port, int type, int proto, int bindFlag);
int SocketClose(SOCKET s);
void SockAddrIN(SOCKADDR_IN *sa, char *addrStr, int port);
SOCKET SocketConnectByName(char *host, int port, int type, int proto, int bindFlag);
int SocketSetLastError();
int SocketGetLastError();
int SocketSetError(int error);

#ifdef __cplusplus
};
#endif

#endif
