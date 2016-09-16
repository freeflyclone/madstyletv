#include "xavfile.h"

XAVFile::XAVFile(const std::string url) :
	XAVSrc(),
	url(url)
{
	xprintf("XAVFile::XAVFile(std::string): '%s'\n", url.c_str());
	XAVSrc::name = url;
	int error;

	if( (error = avformat_open_input(&pFormatCtx, name.c_str(), NULL, NULL)) != 0 )
		throwXAVException("Couldn't open file: "+name);

	if( avformat_find_stream_info(pFormatCtx,NULL) < 0) 
		throwXAVException("Couldn't find stream info: "+name);

	mNumStreams = pFormatCtx->nb_streams;

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
}

XAVFile::~XAVFile() {
}

int XAVFile::Read(uint8_t *buff, int size) {
	return 0;
}

int XAVFile::Write(uint8_t *buff, int size) {
	return 0;
}

int64_t XAVFile::Seek(int64_t offset, int whence) {
	xprintf("Seek not implemented yet: %ld, %d", offset, whence);
	return -1;
}

int XAVFile::read(void *opaque, uint8_t *buff, int size) {
	XAVFile *pXav = (XAVFile *)opaque;
	return pXav->Read(buff, size);
}

int XAVFile::write(void *opaque, uint8_t *buff, int size) {
	XAVFile *pXav = (XAVFile *)opaque;
	return pXav->Write(buff, size);
}

int64_t XAVFile::seek(void *opaque, int64_t offset, int whence) {
	XAVFile *pXav = (XAVFile *)opaque;
	return pXav->Seek(offset, whence);
}
