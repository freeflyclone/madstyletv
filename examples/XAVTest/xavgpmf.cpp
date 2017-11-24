#include "xavgpmf.h"

XAVGpmfThread::XAVGpmfThread(XAVStreamHandle s) : XThread("XAVGpmf" + std::to_string(s->streamIdx)), stream(s), state(0) {
	xprintf("Thread '%s' initializing\n", Name().c_str());
	try{
		pcb = stream->pcb;
		pBuff = new uint8_t[pcb->Size()];
	}
	catch (std::runtime_error e) {
		xprintf("Failed to open file %s: reason: %s\n", Name().c_str(), e.what());
	}
}

void XAVGpmfThread::Run() {
	while (IsRunning()) {
		if (pcb->Count())
			InvokeParsers(pcb);
		else
			std::this_thread::sleep_for(std::chrono::duration<int, std::milli>(10));
	}
}

