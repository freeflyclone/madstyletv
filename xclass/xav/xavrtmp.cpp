#include "xavrtmp.h"

void XAVRtmpThread::logCallback(int level, const char *fmt, va_list arglist) {
	static char buff[2048];
	if(level>RTMP_debuglevel)
		return;
	vsprintf(buff,fmt,arglist);
	strcat(buff,"\n");
	OutputDebugString(buff);
}

XAVRtmpThread::XAVRtmpThread(const std::string url, XFifo *fifo) :
	url(url),
	fifo(fifo)
{
	buffer = new unsigned char[4096];

	if ((pRtmp=RTMP_Alloc()) == NULL)
		throwXAVException("Couldn't allocate an RTMP context");

	RTMP_LogSetLevel(RTMP_LOGINFO);
	RTMP_LogSetCallback(logCallback);
}
	
void *XAVRtmpThread::Run() {
	while(IsRunning()) {
		RTMP_Init(pRtmp);

		if (!RTMP_SetupURL(pRtmp, const_cast<char*>(url.c_str()))) {
			DebugPrintf("URL parse failed: %s", url.c_str());
			Sleep(200);
			continue;
		}

		if( !RTMP_Connect(pRtmp, NULL) ) {
			DebugPrintf("Unable to connect to server: '%s'", url.c_str());
			Sleep(200);
			continue;
		}


		if( !RTMP_ConnectStream(pRtmp, 0) ) {
			DebugPrintf("Unable to connect to stream: '%s'", url.c_str());
			Sleep(200);
			continue;
		}

		DebugPrintf("Server/Stream connection complete: '%s'", url.c_str());

		while(IsRunning()) {
			int nRead=0;
			char buff[4096];
			nRead=RTMP_Read(pRtmp, buff, sizeof(buff));
			if(!nRead){
				DebugPrintf("Server ended connection.  Retrying...");
				RTMP_Close(pRtmp);
				fifo->Flush();
				Sleep(200);
				break;
			}
			fifo->WriteBytes((unsigned char *)buff,nRead);
			Sleep(1);
		}
	}
}

XAVRtmp::XAVRtmp(const std::string url) :
	XAVNet(url),
	thread(url, &fifo)
{
	thread.WaitStart();
}

XAVRtmp::~XAVRtmp() {
}
