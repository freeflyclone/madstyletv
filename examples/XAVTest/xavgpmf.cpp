#include "xavgpmf.h"

XAVGpmfThread::XAVGpmfThread(XAVStreamHandle s) : XThread("XAVGpmf" + std::to_string(s->streamIdx)), stream(s) {
	pcb = stream->pcb;
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
	}

	if (GPMF_VALID_FOURCC(key)) {
		{ // uint32_t aligned, endian agnostic
			uint8_t tmpH, tmpL;
			pcb->Read(&tsl.type, 1);
			pcb->Read(&tsl.size, 1);
			pcb->Read(&tmpH, 1);
			pcb->Read(&tmpL, 1);
			tsl.length = (tmpH << 8) + tmpL;
		}

		if (tsl.type != GPMF_TYPE_NEST) {
			// always read 32bit aligned length
			size_t readCount = (tsl.size * tsl.length + 3) & (~3);

			// early bail-out kludge for 'GPRO' key in stream 4, currently ungroked
			if (readCount > 0x4000) {
				uint32_t skipCount = (tsl.size*tsl.length + 3) & (~3);
				uint64_t remaining = pcb->Skip(skipCount);
				return;
			}

			// there's less than 16K data available, call all registered Listeners
			uint8_t buff[0x4000];
			size_t nRead = pcb->Read(buff, readCount);
			Broadcast(key, tsl, buff);
		}
		else
			Broadcast(key, tsl, nullptr);
	}
}

void XAVGpmfThread::AddListener(uint32_t key, XAVGpmfListener fn){
	listeners[key].push_back(fn);
}

void XAVGpmfThread::AddGenericListener(XAVGpmfListener fn){
	genericListeners.push_back(fn);
}

void XAVGpmfThread::Broadcast(uint32_t key, GPMF_TypeSizeLength tsl, uint8_t* buff){
	for (auto fn : listeners[key])
		fn(key, tsl, buff);

	for (auto fn : genericListeners)
		fn(key, tsl, buff);
}

XAVGpmfTelemetry::XAVGpmfTelemetry() {
	xprintf("XAVGpmfTelemetry::XAVGpmfTelemetry()\n");
}

void XAVGpmfTelemetry::InitListeners(XAVGpmfThreads streams) {
	if (streams.size() < 3)
		throw std::runtime_error("XAVGpmfTelemetry::InitListeners(): not enough streams available");

	listener = [this](uint32_t key, GPMF_TypeSizeLength tsl, uint8_t* buff){
		xprintf("Listener got: %c%c%c%c, %s\n", PRINTF_4CC(key), buff);
	};
	streams[1]->AddListener(STR2FOURCC("STNM"), listener);
}