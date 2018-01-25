#include "xavData.h"

XAVDataThread::XAVDataThread(XAVStreamHandle s) : XThread("XAVData" + std::to_string(s->streamIdx)), stream(s) {
	pcb = stream->pcb;
}

XAVDataThread::~XAVDataThread() {
}

void XAVDataThread::Run() {
	while (IsRunning()) {
		if (pcb->Count()) {
			UpdateStatus("Status Update: " + std::to_string(pcb->Count()));
			for (auto l : listeners) {
				l(this, pcb);
			}
		}
		else
			std::this_thread::sleep_for(std::chrono::duration<int, std::milli>(10));
	}
}

void XAVDataThread::AddListener(XAVDataListener fn){
	if (fn)
		listeners.push_back(fn);
	else
		listeners.clear();
}

std::string XAVDataThread::Status() { 
	std::unique_lock<std::mutex> lock(mutex);
	return status;
}

void XAVDataThread::UpdateStatus(std::string s) {
	std::unique_lock<std::mutex> lock(mutex);
	if (s.size())
		status += s;
	else
		status = "";
}