#ifndef XUART_H
#define XUART_H

#include <string>


class XUart : XThread {
public:
	XUart(std::string portName) : hPort(INVALID_HANDLE_VALUE), XThread(portName+"Thread") {
		std::wstringstream ws;
		ws << portName.c_str();
		std::wstring name = ws.str();

		if ( (hPort = CreateFile(name.c_str(), GENERIC_READ | GENERIC_WRITE, 0, NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL)) == INVALID_HANDLE_VALUE)
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

		Start();
	};

	~XUart(){
		Stop();

		xprintf("XUart::~XUart()\n");
		if (hPort != INVALID_HANDLE_VALUE)
			CloseHandle(hPort);
	};

	void Run() {
		while (IsRunning()) {
			nRead = 0;
			if (ReadFile(hPort, buffer, sizeof(buffer), (LPDWORD)(&nRead), NULL)) {
				xprintf("Read %d bytes\n", nRead);
			}
			else
				std::this_thread::sleep_for(std::chrono::duration<int, std::milli>(1));
		}
	}

	HANDLE hPort;
	DCB portDCB;
	COMMTIMEOUTS   commTimeouts;
	unsigned char buffer[1024];
	unsigned int nRead;
};

#endif