#ifndef XAVGPMF_H
#define XAVGPMF_H

#include <map>
#include <xav.h>
#include <xavsrc.h>
#include <xutils.h>

#include "GPMF_parser.h"

struct GPMF_TypeSizeLength {
	uint8_t type;
	uint8_t size;
	uint16_t length;
};

typedef std::function<void(uint32_t, GPMF_TypeSizeLength, uint8_t*)> XAVGpmfListener;
typedef std::vector<XAVGpmfListener> XAVGpmfListeners;
typedef std::map<uint32_t, XAVGpmfListeners> XAVGpmfListenerList;

class XAVGpmfThread : public XThread {
public:
	XAVGpmfThread(XAVStreamHandle);
	void Run();
	void Parse();

	void AddListener(uint32_t key, XAVGpmfListener fn);
	void Broadcast(uint32_t, GPMF_TypeSizeLength, uint8_t*);

private:
	XAVStreamHandle stream;
	XCircularBuffer *pcb;
	XAVGpmfListenerList listeners;
};

#endif