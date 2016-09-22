#include "xav.h"

XAV::XAV() : XThread("XAV") {
	av_register_all();
}

void XAV::AddSrc(const std::shared_ptr<XAVSrc> src) {
	mSrcs.push_back(src);
}

void XAV::Run() {
	std::vector<std::shared_ptr<XAVSrc> >::iterator i;

	for(i=mSrcs.begin(); i!= mSrcs.end(); i++) 
		i->get()->Start();
		
	for (i = mSrcs.begin(); i != mSrcs.end(); i++)
		i->get()->Stop();

	for (i = mSrcs.begin(); i != mSrcs.end(); i++)
		i->get()->WaitForJoin();

	Stop();
}

std::shared_ptr<XAVSrc> XAV::GetSrc(int idx) {
	return mSrcs[idx];
}
