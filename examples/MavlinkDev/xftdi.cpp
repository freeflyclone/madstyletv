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
		uint32 nWritten;
		uint8 buffer[] = { 0x0B, 0x48, 0x00, 0x25, 0x00, 0x04 };
		FT_STATUS status;

		std::this_thread::sleep_for(std::chrono::duration<int, std::milli>(10));
		status = SPI_Write(pFtdi.ftHandle, buffer, sizeof(buffer), &nWritten, SPI_TRANSFER_OPTIONS_SIZE_IN_BYTES | SPI_TRANSFER_OPTIONS_CHIPSELECT_ENABLE | SPI_TRANSFER_OPTIONS_CHIPSELECT_DISABLE);
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
		xprintf("        ftHandle=0x%x\n", (unsigned int)devList.ftHandle);
	}
	xprintf("Found %d channel%s\n", channels, (channels>1)?"s":"");

	status = SPI_OpenChannel(CHANNEL_TO_OPEN, &ftHandle);
	APP_CHECK_STATUS(status);
	xprintf("handle: %0x%x\n", (unsigned int)ftHandle);

	channelConf.ClockRate = 1000000;
	channelConf.LatencyTimer = 255;
	channelConf.configOptions = SPI_CONFIG_OPTION_MODE0 | SPI_CONFIG_OPTION_CS_DBUS3;
	channelConf.Pin = 0x00000000;

	status = SPI_InitChannel(ftHandle, &channelConf);
	APP_CHECK_STATUS(status);

	/*
	if (FT_CreateDeviceInfoList(&numDevs) != FT_OK)
		throw std::runtime_error("FT_CreateDeviceInfoList() failed");

	if (numDevs < 1)
		throw std::runtime_error("No FTDI devices found");

	xprintf("Found %d devices\n", numDevs);

	if (FT_Open(0, &ftHandle) != FT_OK)
		throw std::runtime_error("Unable to open the first device");

	status = FT_ResetDevice(ftHandle);
	status |= FT_GetQueueStatus(ftHandle, &numBytesToRead);

	if ((status == FT_OK) && (numBytesToRead > 0)) {
		status |= FT_Read(ftHandle, inBuffer, numBytesToRead, &numBytesRead);
	}

	status |= FT_SetUSBParameters(ftHandle, 0x10000, 0xFFFF);
	status |= FT_SetChars(ftHandle, false, 0, false, 0);
	status |= FT_SetTimeouts(ftHandle, 0, 5000);
	status |= FT_SetLatencyTimer(ftHandle, 1);
	status |= FT_SetFlowControl(ftHandle, FT_FLOW_RTS_CTS, 0x00, 0x00);
	status |= FT_SetBitMode(ftHandle, 0x00, 0x00);
	status |= FT_SetBitMode(ftHandle, 0x00, 0x02);

	if (status != FT_OK)
		throw std::runtime_error("Setting up the FTDI for MPSSE didn't work");

	// allow USB to settle
	std::this_thread::sleep_for(std::chrono::duration<int, std::milli>(50));

	numBytesToSend = 0;
	outBuffer[numBytesToSend++] = 0xAB;

	status = FT_Write(ftHandle, outBuffer, numBytesToSend, &numBytesSent);
	numBytesToSend = 0;

	do {
		status = FT_GetQueueStatus(ftHandle, &numBytesToRead);
	} while ((numBytesToRead == 0) && (status == FT_OK));

	bool bCommandEchod = false;
	status = FT_Read(ftHandle, &inBuffer, numBytesToRead, &numBytesRead);
	//Read out the data from input buffer
	for (int dwCount = 0; dwCount < numBytesRead - 1; dwCount++)
		//Check if Bad command and echo command are received
	{
		if ((inBuffer[dwCount] == 0xFA) && (inBuffer[dwCount + 1] == 0xAB))
		{
			bCommandEchod = true;
			break;
		}
	}
	if (bCommandEchod == false)
	{
		FT_Close(ftHandle);
		throw std::runtime_error("Error synchronizing the MPSSE");
	}
	*/
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

void XFtdi::WriteGPIO(uint8_t dir, uint8_t value) {
	status = FT_WriteGPIO(ftHandle, dir, value);
	APP_CHECK_STATUS(status);
	xprintf("WriteGPIO(%02X,%02X)\n", dir, value);
}

void XFtdi::ReadGPIO(uint8_t *value) {
	status = FT_ReadGPIO(ftHandle, value);
	APP_CHECK_STATUS(status);
}