#include "xftdi.h"

// --------------------------
// XFtdi::ReadThread class
// --------------------------
XFtdi::ReadThread::ReadThread(XFtdi &pm) : XThread("XFtdi::ReadThread"), pFtdi(pm) {
	Start();
}

XFtdi::ReadThread::~ReadThread() {
	Stop();
}

// read a byte, pass it to Ftdi_parse_char, if we get a message, 
// call possibly mulitple Listener functions connected to the received msgid 
// and/or possibly multiple Listener functions that receive ALL msgid's
void XFtdi::ReadThread::Run() {
	while (IsRunning())
		std::this_thread::sleep_for(std::chrono::duration<int, std::milli>(100));
}

// ---------------------------
// XFtdi::WriteThread class
// ---------------------------
XFtdi::WriteThread::WriteThread(XFtdi &pm) : XThread("XFtdi::WriteThread"), pFtdi(pm) {
	Start();
}

XFtdi::WriteThread::~WriteThread() {
	Stop();
}

void XFtdi::WriteThread::Run() {
	while (IsRunning()) {
		std::this_thread::sleep_for(std::chrono::duration<int, std::milli>(100));
	}
}

// ---------------------------
// XFtdi main class
// ---------------------------
XFtdi::XFtdi() : rThread(NULL), wThread(NULL), devList({ 0 }), channelConf({ 0 }) {
	uint32 channels;

	xprintf("XFtdi::XFtdi()\n");

	rThread = new ReadThread(*this);
	wThread = new WriteThread(*this);

	Init_libMPSSE();

	status = SPI_GetNumChannels(&channels);
	APP_CHECK_STATUS(status);
	if (channels) {
		status = SPI_GetChannelInfo(0, &devList);
		APP_CHECK_STATUS(status);
		xprintf("Channel Info:\n");
		xprintf("        Flags=0x%x\n", devList.Flags);
		xprintf("        Type=0x%x\n", devList.Type);
		xprintf("        ID=0x%x\n", devList.ID);
		xprintf("        LocId=0x%x\n", devList.LocId);
		xprintf("        SerialNumber=%s\n", devList.SerialNumber);
		xprintf("        Description=%s\n", devList.Description);
		xprintf("        ftHandle=0x%x\n", (unsigned int)devList.ftHandle);/*is 0 unless open*/
	}
	xprintf("Found %d channel%s\n", channels, (channels>1)?"s":"");

	status = SPI_OpenChannel(CHANNEL_TO_OPEN, &ftHandle);
	APP_CHECK_STATUS(status);
	xprintf("handle: %0x%x\n", (unsigned int)ftHandle);
}

XFtdi::~XFtdi() {
	xprintf("XFtdi::~XFtdi()\n");
	if (rThread)
		delete rThread;
	if (wThread)
		delete wThread;

	SPI_CloseChannel(ftHandle);
	Cleanup_libMPSSE();
}

// attach a Listener to a specific msgid
void XFtdi::AddListener(uint8_t msgid, Listener fn) {
	listenersMap[msgid].push_back(fn);
}

// attach a Listener for ALL msgids
void XFtdi::AddListener(Listener fn) {
	listeners.push_back(fn);
}
