#include "xav.h"

XAV::XAV() : XThread("XAV") {
	av_register_all();
	// a URL for a source is unlikely to work without this.
	// It at least is complaining loudly that doing so
	// will be a requirement in the future.
	avformat_network_init();
}

void XAV::AddSrc(const std::shared_ptr<XAVSrc> src) {
	mSrcs.push_back(src);
}

void XAV::Run() {
	for (auto i : mSrcs)
		i.get()->Start();
		
	for (auto i : mSrcs)
		i.get()->Stop();

	for (auto i : mSrcs)
		i.get()->WaitForJoin();

	Stop();
}

std::shared_ptr<XAVSrc> XAV::GetSrc(int idx) {
	return mSrcs[idx];
}
