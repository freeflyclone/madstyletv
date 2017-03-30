#include "xuart.h"

#ifdef WIN32
XUart::XUart(std::string portName) : hPort(INVALID_HANDLE_VALUE) {
	// Windows needs a wide string if we're configured for Unicode, and we are.
	std::wstringstream ws;
	ws << portName.c_str();
	std::wstring name = ws.str();

	if ((hPort = CreateFile(name.c_str(), GENERIC_READ | GENERIC_WRITE, 0, NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL)) == INVALID_HANDLE_VALUE)
		throw std::runtime_error("failed to open serial port: " + portName);

	portDCB.DCBlength = sizeof(DCB);
	GetCommState(hPort, &portDCB);

	portDCB.fBinary = TRUE;                         // Binary mode; no EOF check 
	portDCB.fParity = FALSE;                        // Disable parity checking  
	portDCB.fDsrSensitivity = FALSE;                // DSR sensitivity  
	portDCB.fErrorChar = FALSE;                     // Disable error replacement  
	portDCB.fOutxDsrFlow = FALSE;                   // No DSR output flow control  
	portDCB.fAbortOnError = FALSE;                  // Do not abort reads/writes on error 
	portDCB.fNull = FALSE;                          // Disable null stripping  
	portDCB.BaudRate = 115200;
	portDCB.Parity = NOPARITY;
	portDCB.StopBits = ONESTOPBIT;
	portDCB.ByteSize = 8;

	if (!SetCommState(hPort, &portDCB))
		throw std::runtime_error("SetCommState() failed");

	commTimeouts.ReadIntervalTimeout = 50;
	commTimeouts.ReadTotalTimeoutConstant = 50;
	commTimeouts.ReadTotalTimeoutMultiplier = 10;
	commTimeouts.WriteTotalTimeoutMultiplier = 10;
	commTimeouts.WriteTotalTimeoutConstant = 50;

	if (!SetCommTimeouts(hPort, &commTimeouts))
		throw std::runtime_error("SetCommTimeouts() failed");

	if (!EscapeCommFunction(hPort, SETDTR))
		throw std::runtime_error("EscapeCommFunction(SETDTR) failed");

	if (!EscapeCommFunction(hPort, SETRTS))
		throw std::runtime_error("EscapeCommFunction(SETRTS) failed");
};

XUart::~XUart() {
	if (hPort != INVALID_HANDLE_VALUE)
		CloseHandle(hPort);
};

int XUart::Read(unsigned char *b, int size) {
	int nRead = 0;
	if (ReadFile(hPort, b, size, (LPDWORD)(&nRead), NULL)>0)
		return nRead;
	else {
		unsigned int error = (unsigned int)GetLastError();
		xprintf("ReadFile() failed with: %08X\n", error);
		return 0;
	}
}

int XUart::Write(unsigned char *b, int size) {
	int nWrite;
	if (WriteFile(hPort, b, size, (LPDWORD)(&nWrite), NULL)>0)
		return nWrite;
	else {
		unsigned int error = (unsigned int)GetLastError();
		xprintf("WriteFile() failed with: %08X\n", error);
		return 0;
	}
}
#else
XUart::XUart(std::string portName) : portFd(0) {
    struct termios tty;

    if( (portFd = open(portName.c_str(), O_RDWR | O_NOCTTY | O_SYNC)) < 0)
        throw std::runtime_error("failed to open serial port: "+portName);

    memset (&tty, 0, sizeof tty);

    if (tcgetattr (portFd, &tty) != 0)
        throw std::runtime_error("tcgetattr() failed for port: "+portName);

    cfsetospeed (&tty, B460800);
    cfsetispeed (&tty, B460800);

    tty.c_cflag = (tty.c_cflag & ~CSIZE) | CS8;
    tty.c_iflag &= ~IGNBRK;
    tty.c_lflag = 0;
    tty.c_oflag = 0;
    tty.c_cc[VMIN]  = 1;
    tty.c_cc[VTIME] = 5;
    tty.c_iflag &= ~(IXON | IXOFF | IXANY);
    tty.c_cflag |= (CLOCAL | CREAD);
    tty.c_cflag &= ~(PARENB | PARODD);
    tty.c_cflag |= 0;
    tty.c_cflag &= ~CSTOPB;
    tty.c_cflag &= ~CRTSCTS;

    if (tcsetattr (portFd, TCSANOW, &tty) != 0)
        throw std::runtime_error("tcsetattr() failed for port: "+portName);
}

XUart::~XUart() {
    if (portFd)
        close(portFd);
}

int XUart::Read(unsigned char *b, int size) {
    int nRead = 0;
    if ( (nRead = read(portFd, b, size)) > 0)
        return nRead;
    else {
        xprintf("read() failed with: %08X\n", errno);
        return 0;
    }
}

int XUart::Write(unsigned char *b, int size) {
    int nWrite = 0;
    if ( (nWrite = write(portFd, b, size)) > 0)
        return nWrite;
    else {
        xprintf("write() failed with: %08X\n", errno);
        return 0;
    }
}
#endif
