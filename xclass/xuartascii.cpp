#include "xuartascii.h"

// --------------------------
// XUartAscii::ReadThread class
// --------------------------
XUartAscii::ReadThread::ReadThread(XUartAscii &ps) : XThread("XUartAscii::ReadThread"), pAscii(ps), state(NotSynced) {
    insertPoint = buffer;
	Start();
}

XUartAscii::ReadThread::~ReadThread() {
	Stop();
}

// Assume ASCII text over serial port, delimited by linefeeds
// Sync by initializing to first char after the first linefeed.
// Call Listener functions with each line read.
void XUartAscii::ReadThread::Run() {
	while (IsRunning()) {
		if (pAscii.Read(&cp,1)==0)
			continue;

		switch(state) {
			case NotSynced:
				if(cp == '\n') {
					state = Synced;
					puts("Synced");
				}
				break;

			case Synced:
				if ((insertPoint >= buffer) && (insertPoint <= (buffer+sizeof(buffer)-1)))
					*insertPoint++ = cp;

					if (cp == '\n') {
						*insertPoint++ = 0;
						for (auto fn : pAscii.listeners)
							fn(buffer);
						insertPoint = buffer;
						cp = 0;
					}
				break;
		}
    }
}

// ---------------------------
// XUartAscii::WriteThread class
// ---------------------------
XUartAscii::WriteThread::WriteThread(XUartAscii &ps) : XThread("XUartAscii::WriteThread"), pAscii(ps) {
	Start();
}

XUartAscii::WriteThread::~WriteThread() {
	Stop();
}

void XUartAscii::WriteThread::Run() {
	while (IsRunning()) {
		std::this_thread::sleep_for(std::chrono::duration<int, std::milli>(100));
	}
}

// ---------------------------
// XUartAscii main class
// ---------------------------
XUartAscii::XUartAscii(std::string n) : XUart(n), rThread(NULL), wThread(NULL) {
	xprintf("XUartAscii::XUartAscii()\n");

	rThread = new ReadThread(*this);
	wThread = new WriteThread(*this);
}

XUartAscii::~XUartAscii() {
	xprintf("XUartAscii::~XUartAscii()\n");
	if (rThread)
		delete rThread;
	if (wThread)
		delete wThread;
}

void XUartAscii::AddListener(Listener fn) {
	listeners.push_back(fn);
}
