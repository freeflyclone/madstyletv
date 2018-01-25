#include "xavgpmf.h"

XAVGpmfThread::XAVGpmfThread(XAVStreamHandle s) : XThread("XAVGpmf" + std::to_string(s->streamIdx)), stream(s) {
	pcb = stream->pcb;
	if ((pBuff = new uint8_t[pcb->Size()]) == nullptr)
		throw std::runtime_error(Name() + ": Unable to able to allocate pBuff");
}

XAVGpmfThread::~XAVGpmfThread() {
	delete pBuff;
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
			tsl.count = (tmpH << 8) + tmpL;
		}

		if (tsl.type != GPMF_TYPE_NEST) {
			// always read 32bit aligned length
			size_t readCount = (tsl.size * tsl.count + 3) & (~3);

			// early bail-out kludge for 'GPRO' key in stream 4, currently ungroked
			if (readCount >= pcb->Count()) {
				uint32_t skipCount = (tsl.size*tsl.count + 3) & (~3);
				pcb->Skip(skipCount);
				return;
			}

			// there's less than XCircularBufferDefaultSize  data available, call all registered Listeners
			size_t nRead = pcb->Read(pBuff, readCount);
			Broadcast(key, tsl, pBuff);
		}
		else
			Broadcast(key, tsl, nullptr);
	}
}

void XAVGpmfThread::AddListener(uint32_t key, XAVGpmfListener fn){
	listeners[key].push_back(fn);
}

void XAVGpmfThread::AddGenericListener(XAVGpmfListener fn){
	if (fn == nullptr)
		genericListeners.clear();
	else
		genericListeners.push_back(fn);
}

void XAVGpmfThread::Broadcast(uint32_t key, GPMF_TypeSizeLength tsl, uint8_t* buff){
	for (auto fn : genericListeners)
		fn(key, tsl, buff);

	for (auto fn : listeners[key])
		fn(key, tsl, buff);
}

XAVGpmfTelemetry::XAVGpmfTelemetry() {
	xprintf("XAVGpmfTelemetry::XAVGpmfTelemetry()\n");
}

void XAVGpmfTelemetry::InitListeners(XAVGpmfThreads streams) {
	if (streams.size() < 2)
		throw std::runtime_error("XAVGpmfTelemetry::InitListeners(): not enough streams available");

	XAVGpmfListener fn = [this](uint32_t key, GPMF_TypeSizeLength tsl, uint8_t* buff){ PrintGPMF(key, tsl); };
	streams[1]->AddGenericListener(fn);
}

void XAVGpmfTelemetry::PrintGPMF(uint32_t key, GPMF_TypeSizeLength tsl) {
	if (tsl.type == GPMF_TYPE_NEST)
		xprintf("%c%c%c%c: type %d, size: %02X, count: %d\n", PRINTF_4CC(key), tsl.type, tsl.size, tsl.count);
	else
		xprintf("%c%c%c%c: type '%c', size: %02X, count: %d\n", PRINTF_4CC(key), tsl.type, tsl.size, tsl.count);
}

std::string XAVGpmfTelemetry::Status() { 
	std::lock_guard<std::mutex> lock(mutex);
	return status; 
}

void XAVGpmfTelemetry::UpdateStatus(std::string s) { 
	std::lock_guard<std::mutex> lock(mutex);
	if (s.size())
		status += s; 
	else 
		status = ""; 
}
