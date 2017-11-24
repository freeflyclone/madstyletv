#ifndef XAVGPMF_H
#define XAVGPMF_H

#include <xav.h>
#include <xavsrc.h>
#include <xutils.h>

#include "GPMF_parser.h"

typedef std::function<void(XCircularBuffer *)> XAVGpmfParser;
typedef std::vector<XAVGpmfParser> XAVGpmfParsers;

class XAVGpmfThread : public XThread {
public:
	XAVGpmfThread(XAVStreamHandle);
	void Run();

	void AddParser(XAVGpmfParser fn) {
		parsers.emplace_back(fn);
	}

	void InvokeParsers(XCircularBuffer *pcb) {
		for (auto fn : parsers)
			fn(pcb);
	}

	XAVStream* Stream() { return stream.get(); }

//private:
	XAVStreamHandle stream;
	uint8_t *pBuff;
	XCircularBuffer *pcb;
	XAVGpmfParsers parsers;

	uint32_t state;
};

#endif