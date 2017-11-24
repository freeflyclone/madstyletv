#include "xavgpmf.h"

XAVGpmfThread::XAVGpmfThread(XAVStreamHandle s) : XThread("XAVGpmf" + std::to_string(s->streamIdx)), stream(s) {
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
			Parse();
		else
			std::this_thread::sleep_for(std::chrono::duration<int, std::milli>(10));
	}
}

void XAVGpmfThread::Parse() {
	uint32_t streamId = stream->streamIdx;
	uint32_t key;
	GPMF_TypeSizeLength tsl;

	pcb->Read((uint8_t*)&key, sizeof(key));
	if (key == STR2FOURCC("GPRO")) {
		int skipped = 0;
		do {
			pcb->Read((uint8_t*)&key, sizeof(key));
			skipped += 4;
		} while (key != STR2FOURCC("DEVC"));
		xprintf("Stream(%d) - Skipped %d bytes\n", streamId, skipped);
	}

	if (GPMF_VALID_FOURCC(key)) {
		pcb->Read(&tsl.type, 1);
		pcb->Read(&tsl.size, 1);
		pcb->Read((uint8_t*)&tsl.length, 2);
		tsl.length = BYTESWAP16(tsl.length);

		if (GPMF_TYPE_NEST == tsl.type) {
			xprintf("Stream(%d): %c%c%c%c\n", streamId, PRINTF_4CC(key));
			Broadcast(key, tsl, nullptr);
		}
		else if (tsl.type != GPMF_TYPE_NEST) {
			// always read 32-bit aligned length
			uint32_t bytesToRead = (tsl.size*tsl.length + 3) & (~3);

			// kludgey limit because 'GPRO' (streamId 4) ends badly, and I'm 
			// not interested in this particular rabbit hole right now.
			if (bytesToRead < 0x4000) {
				xprintf("Stream(%d): %c%c%c%c, '%c',%02X,%04X\n", streamId, PRINTF_4CC(key), tsl.type, tsl.size, tsl.length);

				// tmp buff on stack to eliminate new/delete
				uint8_t buff[0x4000];
				uint64_t nRead = pcb->Read(buff, bytesToRead);
				Broadcast(key, tsl, buff);
			}
		}
	}
}

void XAVGpmfThread::AddListener(XAVGpmfListener fn){
	listeners.emplace_back(fn);
}

void XAVGpmfThread::Broadcast(uint32_t key, GPMF_TypeSizeLength tsl, uint8_t* buff){
	for (auto fn : listeners)
		fn(key, tsl, buff);
}

