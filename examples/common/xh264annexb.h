#ifndef XH264ANNEXB_H
#define XH264ANNEXB_H

extern "C" {
	#include "h264decoder.h"
	#include "annexb.h"
};

class Xh264AnnexB {
public:
	int  GetNALU(VideoParameters *p_Vid, NALU_t *nalu, ANNEXB_t *annex_b);
	void Open(char *fn, ANNEXB_t *annex_b);
	void Close(ANNEXB_t *annex_b);
	void Malloc(VideoParameters *p_Vid, ANNEXB_t **p_annex_b);
	void Free(ANNEXB_t **p_annex_b);
	void Init(ANNEXB_t *annex_b);
	void Reset(ANNEXB_t *annex_b);
};

#endif