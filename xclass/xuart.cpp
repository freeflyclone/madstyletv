#include "xuart.h"

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
	portDCB.fParity = TRUE;                         // Enable parity checking  
	portDCB.fDsrSensitivity = FALSE;                // DSR sensitivity  
	portDCB.fErrorChar = FALSE;                     // Disable error replacement  
	portDCB.fOutxDsrFlow = FALSE;                   // No DSR output flow control  
	portDCB.fAbortOnError = FALSE;                  // Do not abort reads/writes on error 
	portDCB.fNull = FALSE;                          // Disable null stripping  
	portDCB.fTXContinueOnXoff = TRUE;               // XOFF continues Tx  
	portDCB.BaudRate = 115200;
	portDCB.Parity = NOPARITY;
	portDCB.StopBits = ONESTOPBIT;

	if (!SetCommState(hPort, &portDCB))
		throw std::runtime_error("SetCommState() failed");

	commTimeouts.ReadIntervalTimeout = 50;
	commTimeouts.ReadTotalTimeoutConstant = 50;
	commTimeouts.ReadTotalTimeoutMultiplier = 10;
	commTimeouts.WriteTotalTimeoutMultiplier = 10;
	commTimeouts.WriteTotalTimeoutConstant = 50;

	if (!SetCommTimeouts(hPort, &commTimeouts))
		throw std::runtime_error("SetCommTimeouts() failed");
};

XUart::~XUart() {
	xprintf("XUart::~XUart()\n");
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
