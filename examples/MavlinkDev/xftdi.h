/*
** XFtdi - a class for managing an asynchronous calls to an FTDI
** serial converter, in this case USB <-> SPI.
**
** The class supports adding so-called Listener functions as needed
** (using C++ lambda functions) to provide higher level functionality,
** such as using sensor data, mission planning, vehicle configuration, etc.
*/
#ifndef XFTDI_H
#define XFTDI_H

#include "xclasses.h"

#include "ftd2xx.h"
#include "libMPSSE_spi.h"

#define APP_CHECK_STATUS(exp) {if(exp!=FT_OK){printf("%s:%d:%s(): status(0x%x) \
!= FT_OK\n",__FILE__, __LINE__, __FUNCTION__,exp);exit(1);}else{;}};

#define CHECK_NULL(exp){if(exp==NULL){printf("%s:%d:%s():  NULL expression \
encountered \n",__FILE__, __LINE__, __FUNCTION__);exit(1);}else{;}};

#define SPI_DEVICE_BUFFER_SIZE		256
#define SPI_WRITE_COMPLETION_RETRY		10
#define START_ADDRESS_EEPROM 	0x00 /*read/write start address inside the EEPROM*/
#define END_ADDRESS_EEPROM		0x10
#define RETRY_COUNT_EEPROM		10	/* number of retries if read/write fails */
#define CHANNEL_TO_OPEN			0	/*0 for first available channel, 1 for next... */
#define SPI_SLAVE_0				0
#define SPI_SLAVE_1				1
#define SPI_SLAVE_2				2
#define DATA_OFFSET				4
#define USE_WRITEREAD			0

class XFtdi : public XObject {
public:
	// A function type that accepts a MAVLINK message.
	typedef std::function<void()> Listener;
	// A list of functions that allows calling multiple functions per message.
	typedef std::vector<Listener> Listeners;
	// A map of function lists, sorted by msgid, that allows calling multiple functions only for their registered msgid.
	typedef std::map<uint8_t, Listeners> ListenersMap;

	class ReadThread : public XThread {
	public:
		ReadThread(XFtdi &);
		~ReadThread();
		void Run();

	private:
		unsigned char cp, parseState;
		XFtdi &pFtdi;
	};

	class WriteThread : public XThread {
	public:
		WriteThread(XFtdi &);
		~WriteThread();
		void Run();
		bool WriteMessage();

	private:
		XFtdi &pFtdi;
	};

	XFtdi();
	~XFtdi();
	void AddListener(uint8_t, Listener);
	void AddListener(Listener);

	void WriteGPIO(uint8_t dir, uint8_t value);
	void ReadGPIO(uint8_t *value);

private:
	ReadThread *rThread;
	WriteThread *wThread;
	Listeners listeners;
	ListenersMap listenersMap;

	FT_STATUS status;
	FT_DEVICE_LIST_INFO_NODE devList;
	ChannelConfig channelConf;
	FT_HANDLE ftHandle;
	uint8 buffer[256];
};


#endif