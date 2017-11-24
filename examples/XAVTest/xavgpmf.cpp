#include "xavgpmf.h"

XAVGpmfThread::XAVGpmfThread(XAVStreamHandle s) : XThread("XAVGpmf" + std::to_string(gpmfStreamId++)), stream(s) {
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
		size_t nRead = pcb->Read(pBuff, pcb->Count());
		if (nRead) {
			//xprintf("Stream: %d, %d bytes, %d\n", stream->streamIdx, nRead, pcb->Count());
			InvokeParsers(pBuff, nRead);
		}
		else
			std::this_thread::sleep_for(std::chrono::duration<int, std::milli>(1));
	}
}

