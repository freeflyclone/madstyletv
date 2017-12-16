#ifndef XAVGPMF_H
#define XAVGPMF_H

#include <map>
#include <xav.h>
#include <xavsrc.h>
#include <xcircularbuffer.h>
#include <xutils.h>

#include "GPMF_parser.h"

struct GPMF_TypeSizeLength {
	uint8_t type;
	uint8_t size;
	uint16_t count;
};

typedef std::function<void(uint32_t, GPMF_TypeSizeLength, uint8_t*)> XAVGpmfListener;

typedef std::vector<XAVGpmfListener> XAVGpmfListeners;
typedef std::map<uint32_t, XAVGpmfListeners> XAVGpmfListenerList;

class XAVGpmfThread;
typedef std::vector<XAVGpmfThread *> XAVGpmfThreads;

class XAVGpmfThread : public XThread {
public:
	XAVGpmfThread(XAVStreamHandle);
	virtual ~XAVGpmfThread();

	void Run();
	void Parse();

	void AddListener(uint32_t key, XAVGpmfListener fn);
	void AddGenericListener(XAVGpmfListener fn);
	void Broadcast(uint32_t, GPMF_TypeSizeLength, uint8_t*);

private:
	XAVStreamHandle stream;				// XAV wrapper around various FFMpeg things
	XCircularBuffer *pcb;				// points to the one in 'stream'
	XAVGpmfListenerList listeners;		// callback functions per GPMF key
	XAVGpmfListeners genericListeners;	// callback functions for all keys
	uint8_t *pBuff;						// for pcb->Read() use
};

class XAVGpmfTelemetry {
public:
	XAVGpmfTelemetry();

	void InitListeners(XAVGpmfThreads);
	void PrintGPMF(uint32_t key, GPMF_TypeSizeLength tsl);

	std::string Status();
	void UpdateStatus(std::string);
	std::string status;
	std::mutex mutex;
};

#endif