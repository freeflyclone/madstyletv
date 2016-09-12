#include "xavnet.h"

// 2MB buffer size
static int bufferSize = 0x200000;

XAVNet::XAVNet(const std::string url) :
	XAVSrc(),
	url(url),
	buffer(NULL),
	fifo(bufferSize*2),
	pAvioCtx(NULL)
{
	DebugPrintf("XAVNet::XAVNet(std::string): '%s'", url.c_str());
	XAVSrc::name = url;

	if ( (buffer=(unsigned char *)av_malloc(bufferSize)) == NULL)
		throwXAVException("Couldn't allocate a buffer");

	if( (pAvioCtx = avio_alloc_context(buffer, bufferSize, 0, this, read, write, NULL)) == NULL)
		throwXAVException("Couldn't allocate an AVIOContext");

	if( (pFormatCtx = avformat_alloc_context()) == NULL )
		throwXAVException("Couldn't allocate an AVFormatContext");

	pFormatCtx->pb = pAvioCtx;
}

XAVNet::~XAVNet() {
	if (pAvioCtx)
		av_free(pAvioCtx);
	if (buffer)
		av_free(buffer);

	buffer = NULL;
	pAvioCtx = NULL;
}

void *XAVNet::Run() {
	bool formatKnown = false;

	while(IsRunning()) {
		if( !formatKnown )
		{
			int fullness = fifo.Count();
			int probeSize = 16384*4;
			if(fullness < probeSize)
				continue;

			DebugPrintf("Fifo has at least %d bytes", probeSize);

			int nRead = fifo.ReadBytes(detectBuffer,probeSize);
			fifo.Rewind(probeSize);

			AVProbeData pd;
			pd.buf = detectBuffer;
			pd.buf_size = probeSize;
			pd.filename = "";
			pFormatCtx->iformat = av_probe_input_format(&pd, 1);
			pFormatCtx->flags = AVFMT_FLAG_CUSTOM_IO;

			if(avformat_open_input(&pFormatCtx, "", NULL, NULL)!=0) {
				//throwXAVException("Couldn't open src");
				DebugPrintf("Couldn't open src");
				continue;
			}

			if( avformat_find_stream_info(pFormatCtx,NULL) < 0) {
				//throwXAVException("Couldn't find stream info");
				DebugPrintf("Couldn't find stream info");
				continue;
			}

			mNumStreams = pFormatCtx->nb_streams;
			DebugPrintf("numStreams: %d", pFormatCtx->nb_streams);

			for(int i=0; i<mNumStreams; i++) {
				if ((pFormatCtx->streams[i]->codec->codec_type == AVMEDIA_TYPE_VIDEO) && !mVideoStream) {
					mVideoStream = std::make_shared<XAVStream>(pFormatCtx->streams[i]->codec);
					mStreams.emplace_back(mVideoStream);
				}
				else if ((pFormatCtx->streams[i]->codec->codec_type == AVMEDIA_TYPE_AUDIO) && !mAudioStream) {
					mAudioStream = std::make_shared<XAVStream>(pFormatCtx->streams[i]->codec);
					mStreams.emplace_back(mAudioStream);
				} 
			}

			formatKnown = true;
		}

		if( fifo.Count() ) {
			while( av_read_frame(pFormatCtx, &packet) >= 0 ) {
				mStreams[packet.stream_index]->Decode(&packet);
				av_free_packet(&packet);
			}
		}
		else
			Sleep(1);
	}
}

int XAVNet::Read(uint8_t *buff, int size) {
	return fifo.ReadBytes(buff, size);
}

int XAVNet::Write(uint8_t *buff, int size) {
	return fifo.WriteBytes(buff, size);
}

int64_t XAVNet::Seek(int64_t offset, int whence) {
	DebugPrintf("Seek not implemented yet: %ld, %d", offset, whence);
	return -1;
}

int XAVNet::read(void *opaque, uint8_t *buff, int size) {
	XAVNet *pXav = (XAVNet *)opaque;
	return pXav->Read(buff, size);
}

int XAVNet::write(void *opaque, uint8_t *buff, int size) {
	XAVNet *pXav = (XAVNet *)opaque;
	return pXav->Write(buff, size);
}

int64_t XAVNet::seek(void *opaque, int64_t offset, int whence) {
	XAVNet *pXav = (XAVNet *)opaque;
	return pXav->Seek(offset, whence);
}
