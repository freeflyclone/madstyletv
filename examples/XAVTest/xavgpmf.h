#ifndef XAVGPMF_H
#define XAVGPMF_H

#include <xav.h>
#include <xavsrc.h>
#include <xutils.h>

typedef std::function<void(uint8_t *, size_t)> XAVGpmfParser;
typedef std::vector<XAVGpmfParser> XAVGpmfParsers;

class XAVGpmfThread : public XThread {
public:
	XAVGpmfThread(XAVStreamHandle);
	void Run();

	void AddParser(XAVGpmfParser fn) {
		parsers.emplace_back(fn);
	}

	void InvokeParsers(uint8_t* b, size_t s) {
		for (auto fn : parsers)
			fn(b, s);
	}

	XAVStream* Stream() { return stream.get(); }

private:
	XAVStreamHandle stream;
	uint8_t *pBuff;
	XCircularBuffer *pcb;
	FILE *f;
	XAVGpmfParsers parsers;
	int gpmfStreamId = { 0 };
};

#endif