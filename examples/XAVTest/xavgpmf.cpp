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
	if (streams.size() < 3)
		throw std::runtime_error("XAVGpmfTelemetry::InitListeners(): not enough streams available");

	listener = [this](uint32_t key, GPMF_TypeSizeLength tsl, uint8_t* buff){
		if (STR2FOURCC("STNM") == key) {
			xprintf("%c%c%c%c, %s\n", PRINTF_4CC(key), buff);
		}
		else if (STR2FOURCC("ACCL") == key) {
			xprintf("ACCL: type '%c', size: %02X, count: %d\n", tsl.type, tsl.size, tsl.count);
		}
		else if (STR2FOURCC("GYRO") == key) {
			xprintf("GYRO: type '%c', size: %02X, count: %d\n", tsl.type, tsl.size, tsl.count);
		}
		else if (STR2FOURCC("MAGN") == key) {
			xprintf("MAGN: type '%c', size: %02X, count: %d\n", tsl.type, tsl.size, tsl.count);
		}
		else if (STR2FOURCC("TSMP") == key) {
			if (false) {
				uint32_t nSamples{ 0 };
				memcpy((uint8_t*)&nSamples, buff, 4);
				nSamples = BYTESWAP32(nSamples);
				xprintf("%c%c%c%c, '%c', size: %02X, count: %d\n", PRINTF_4CC(key), tsl.type, tsl.size, tsl.count, nSamples);
			}
		}
		else {
			if (tsl.type == 0)
				;//xprintf("%c%c%c%c %d, %d, %d\n", PRINTF_4CC(key), tsl.type, tsl.size, tsl.count);
			else
				;//xprintf("%c%c%c%c, '%c' %02X, %04X\n", PRINTF_4CC(key), tsl.type, tsl.size, tsl.count);
		}
	};
	streams[0]->AddGenericListener(listener);
	streams[1]->AddGenericListener(listener);
	streams[2]->AddGenericListener(listener);
}