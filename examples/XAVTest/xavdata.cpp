#include "xavData.h"

XAVDataThread::XAVDataThread(XAVStreamHandle s) : XThread("XAVData" + std::to_string(s->streamIdx)), stream(s) {
	pcb = stream->pcb;
}

XAVDataThread::~XAVDataThread() {
}

void XAVDataThread::Run() {
	while (IsRunning()) {
		if (pcb->Count()) {
			for (auto l : listeners) {
				l(this, pcb);
			}
		}
		else
			std::this_thread::sleep_for(std::chrono::duration<int, std::milli>(10));
	}
}

void XAVDataThread::AddListener(XAVDataListener fn){
	listeners.push_back(fn);
}

