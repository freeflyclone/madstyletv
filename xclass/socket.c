// from MobiusDigitalVideo source code
#include "socket.h"
#include <stdlib.h>
#include <stdio.h>

#include "xutils.h"

unsigned tlsSockError;

static char hostAddrStr[MAX_PATH] = "\0";
static char hostName[MAX_PATH] = "\0";

char *SocketHostToAddr(char *host)
{
	struct hostent *pHE = NULL;
	static char addrStr[MAX_PATH];

	if( !host || strlen(host) == 0 )
		return NULL;

	if ((pHE = gethostbyname(host)) != NULL)
	{
		unsigned char *s = pHE->h_addr;

		s = pHE->h_addr;
		sprintf(hostAddrStr, "%d.%d.%d.%d", s[0], s[1], s[2], s[3]);
		return hostAddrStr;
	}

	return NULL;
}

char *SocketHostAddr()
{
	struct hostent *pHE;
	unsigned char *s;

	if( strlen(hostName) == 0 )
		gethostname(hostName, sizeof(hostName));

	if( strlen(hostAddrStr) == 0 )
	{
		pHE = gethostbyname(hostName);

		if( pHE )
		{
			s = pHE->h_addr;
			sprintf(hostAddrStr, "%d.%d.%d.%d",s[0],s[1],s[2],s[3]);
		}
	}
	return hostAddrStr;
}

char *SocketHostName()
{
	if( strlen(hostName) == 0 )
		gethostname(hostName, sizeof(hostName));

	return hostName;
}

void SockAddrIN(SOCKADDR_IN *sa, char *addrStr, int port)
{
	int a1, a2, a3, a4;

	memset(sa, 0, sizeof(struct sockaddr_in));
	sa->sin_family = AF_INET;
	sa->sin_port = htons((unsigned short)port);
	if( addrStr )
	{
		sscanf(addrStr, "%d.%d.%d.%d",&a1,&a2,&a3,&a4);
		sa->sin_addr.S_un.S_un_b.s_b1 = a1;
		sa->sin_addr.S_un.S_un_b.s_b2 = a2;
		sa->sin_addr.S_un.S_un_b.s_b3 = a3;
		sa->sin_addr.S_un.S_un_b.s_b4 = a4;
	}
	else
		sa->sin_addr.s_addr = INADDR_ANY;
}

int SocketsSetup()
{
	static char isInitialized = 0;

	if (!isInitialized) {
		WORD wVersionRequested;
		int err;
		WSADATA wsaData;

		wVersionRequested = MAKEWORD(2, 2);

		if ((err = WSAStartup(wVersionRequested, &wsaData)) != 0)
		{
			xprintf("SocketsSetup() WSAStartup() failed: %d\n", WSAGetLastError());
			return INVALID_SOCKET;
		}

		if (LOBYTE(wsaData.wVersion) == 2 && HIBYTE(wsaData.wVersion) == 2) {
			isInitialized = 1;
			return 0;
		}

		WSACleanup();
		return INVALID_SOCKET;
	}
	return 0;
}

SOCKET SocketOpen(char *addrStr, int port, int type, int proto, int bindFlag)
{
	SOCKET sock;
	struct sockaddr_in sAddr;
	int bOpt = TRUE;
	int one = 1;
	int reuse = 1;
	short ttl = 7;
	int bc = 1;
	struct ip_mreq  imr;
	int value = 0;
	int valueSize = sizeof(value);
	 
	/* Initialize SOCKADDR_IN structure with destination address and port */
	if( addrStr && strlen(addrStr) )
		SockAddrIN(&sAddr, addrStr, port);
	else
		SockAddrIN(&sAddr, SocketHostAddr(), port);

	/* create a socket... */
	if( (sock = socket(AF_INET, type, proto)) == INVALID_SOCKET )
	{
		xprintf("Can't create socket: %d\n", WSAGetLastError());
		WSACleanup( );
		return INVALID_SOCKET; 
	}

	/* If it's a multicast address...*/
	if( sAddr.sin_addr.s_net >= 224 && sAddr.sin_addr.s_net <= 240 )
	{
		sAddr.sin_addr.s_addr = INADDR_ANY;
		if( setsockopt(sock, SOL_SOCKET, SO_REUSEADDR, (char *) &reuse, sizeof(reuse)) == SOCKET_ERROR )
			xprintf("SocketOpen(%s:%d) multicast setsockopt SO_REUSEADDR failed: %d\n", addrStr, port, WSAGetLastError());

		/* bind() the socket to that address */
		if( (bind(sock, (SOCKADDR *)&sAddr, sizeof(sAddr))) != 0 )
		{
			xprintf("Unable to bind multicast socket %s:%d: %d\n",addrStr,port,WSAGetLastError());
			WSACleanup();
			return INVALID_SOCKET;
		}

		/* setup MULTICAST group membership for the socket */
		SockAddrIN(&sAddr, addrStr, port);
		imr.imr_multiaddr.s_addr = sAddr.sin_addr.s_addr;
		imr.imr_interface.s_addr = INADDR_ANY;
		if (setsockopt(sock, IPPROTO_IP, IP_ADD_MEMBERSHIP, (char *) &imr, sizeof(struct ip_mreq)) != 0) 
		{
			xprintf("setsockopt IP_ADD_MEMBERSHIP: %d\n",WSAGetLastError());
			return 0;
		}

		/* Set the TTL to some initial value.  TBD: What about admin scoping? */
		if (setsockopt(sock, IPPROTO_IP, IP_MULTICAST_TTL, (char *) &ttl, sizeof(ttl)) != 0) 
		{
			xprintf("SocketOpen(%s:%d) setsockopt IP_MULTICAST_TTL failed: %d\n", addrStr, port, WSAGetLastError());
			return 0;
		}
	}
	/* else it's a unicast address, might be local NIC address */
	else
	{
		sAddr.sin_addr.s_addr = INADDR_ANY;
		if( setsockopt(sock, SOL_SOCKET, SO_REUSEADDR, (char *) &reuse, sizeof(reuse)) == SOCKET_ERROR )
			xprintf("SocketOpen(%s:%d) unicast setsockopt SO_REUSEADDR failed: %d\n", addrStr, port, WSAGetLastError());
		if( bindFlag )
		{
			if( (bind(sock, (SOCKADDR *)&sAddr, sizeof(sAddr))) != 0 )
			{
				xprintf("Unable to bind unicast socket %s:%d: %d\n",addrStr,port,WSAGetLastError());
				WSACleanup();
				return INVALID_SOCKET;
			}
		}
	}

	/* get a nice big recieve and send buffer */
	value = 0x10000;
	if( (setsockopt(sock, SOL_SOCKET, SO_RCVBUF, (char *)&value, valueSize)) != 0)
	{
		xprintf("setsockopt(SO_RCVBUF) failed\n");
		return 0;
	}
	if( (setsockopt(sock, SOL_SOCKET, SO_SNDBUF, (char *)&value, valueSize)) != 0)
	{
		xprintf("setsockopt(SO_SNDBUF) failed\n");
		return 0;
	}
	return sock;
}

int SocketClose(SOCKET s)
{
	return closesocket(s);
}

SOCKET SocketConnectByName(char *host, int port, int type, int proto, int bindFlag)
{
	struct hostent *pHE;
	unsigned char hostAddrStr[MAX_PATH], *s;
	SOCKADDR_IN sa;
	SOCKET sock;

	// can't very well connect by name if no name is specified, now can we??
	if( !host || !strlen(host) )
	{
		SocketSetError(WSAEINVAL );
		return INVALID_SOCKET;
	}

	// use the resolver to get the IP address of the named host
	pHE = gethostbyname(host);
	if( !pHE )
	{
		xprintf("SocketConnectByName() couldn't get hostent for '%s'\n", host);
		SocketSetLastError();
		return INVALID_SOCKET;
	}

	// build a SOCKADDR_IN structure if the name was resolved
	s = pHE->h_addr;
	sprintf(hostAddrStr, "%d.%d.%d.%d",s[0],s[1],s[2],s[3]);
	SockAddrIN(&sa, hostAddrStr, port);

	// since we're "CONNECT" by name, assume SOCK_STREAM and TCP socket, and try to create a socket
	if( (sock = SocketOpen(NULL, 0, SOCK_STREAM, IPPROTO_TCP, FALSE)) == INVALID_SOCKET )
	{
		SocketSetLastError();
		return INVALID_SOCKET;
	}

	// attempt to connect to the remote end
	if( connect(sock, (SOCKADDR *)&sa, sizeof(sa)) ==  SOCKET_ERROR )
	{
		SocketSetLastError();
		SocketClose(sock);
		return INVALID_SOCKET;
	}

	return sock;
}

SocketSetError(int error)
{
	TlsSetValue(tlsSockError, (LPVOID)error);
	return error;
}
SocketSetLastError()
{
	int error = WSAGetLastError();
	TlsSetValue(tlsSockError, (LPVOID)error);
	return error;
}
SocketGetLastError()
{
	return (int)TlsGetValue(tlsSockError);
}
