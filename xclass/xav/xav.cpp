#include "xav.h"

XAV::XAV()
{
	if( WSAStartup(MAKEWORD(2,2),&wsaData) != 0)
		throwXAVException("WSAStartup() failed");

	av_register_all();
}

void XAV::AddSrc(const std::shared_ptr<XAVSrc> src) {
	mSrcs.push_back(src);
}

void XAV::Run() {
	std::vector<std::shared_ptr<XAVSrc> >::iterator i;

	for(i=mSrcs.begin(); i!= mSrcs.end(); i++) 
		i->get()->WaitStart();
	
	
	for(i=mSrcs.begin(); i!= mSrcs.end(); i++)
		i->get()->Wait();

	return (void *)1;
}

std::shared_ptr<XAVSrc> XAV::GetSrc(int idx) {
	return mSrcs[idx];
}
