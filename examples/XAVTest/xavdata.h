#ifndef XAVDATA_H
#define XAVDATA_H

#include <map>
#include <xav.h>
#include <xavsrc.h>
#include <xcircularbuffer.h>
#include <xutils.h>

typedef std::function<void(void*, XCircularBuffer*)> XAVDataListener;

typedef std::vector<XAVDataListener> XAVDataListeners;

class XAVDataThread;
typedef std::vector<XAVDataThread *> XAVDataThreads;

class XAVDataThread : public XThread {
public:
	XAVDataThread(XAVStreamHandle);
	virtual ~XAVDataThread();

	void Run();

	void AddListener(XAVDataListener fn);

	// thread-safe status retrieval & update
	std::string Status();
	void UpdateStatus(std::string);		// accumulate into 'status', empty input empties 'status'

private:
	XAVStreamHandle stream;				// XAV wrapper around various FFMpeg things
	XCircularBuffer *pcb;				// points to the one in 'stream'
	XAVDataListeners listeners;			// callback functions for further parsing
	std::string status;
	std::mutex mutex;
};

#endif