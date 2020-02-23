#ifndef XH264ANNEXB_H
#define XH264ANNEXB_H

extern "C" {
	#include "h264decoder.h"
	#include "annexb.h"
};

class Xh264AnnexB {
public:
	virtual int  GetNALU(VideoParameters *p_Vid, NALU_t *nalu, ANNEXB_t *annex_b) = 0;
	virtual void Open(char *fn, ANNEXB_t *annex_b) = 0;
	virtual void Close(ANNEXB_t *annex_b) = 0;
	virtual void Malloc(VideoParameters *p_Vid, ANNEXB_t **p_annex_b) = 0;
	virtual void Free(ANNEXB_t **p_annex_b) = 0;
	virtual void Init(ANNEXB_t *annex_b) = 0;
	virtual void Reset(ANNEXB_t *annex_b) = 0;
};

#endif