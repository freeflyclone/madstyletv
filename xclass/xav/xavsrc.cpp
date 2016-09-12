#include "xavsrc.h"

XAVStream::XAVStream(AVCodecContext *ctx) :
		freeBuffs(1,1),
		usedBuffs(0,1)
{
	pCodecCtx = ctx;

	pCodec=avcodec_find_decoder(pCodecCtx->codec_id);
	if(pCodec==NULL)
		throwXAVException("codec not found");

	if( avcodec_open2(pCodecCtx, pCodec,NULL) < 0 )
		throwXAVException("Failed to open codec");

	if( pCodecCtx->codec_type == AVMEDIA_TYPE_VIDEO ) {
		DebugPrintf("Found AVMEDIA_TYPE_VIDEO");
		DebugPrintf("                   width: %d", pCodecCtx->width);
		DebugPrintf("                  height: %d", pCodecCtx->height);
		pFrame = av_frame_alloc();
		numBytes = avpicture_get_size(pCodecCtx->pix_fmt, pCodecCtx->width, pCodecCtx->height) * 4;
		buffer = (unsigned char *)av_malloc(numBytes*sizeof(unsigned char));

		memset(buffer, 0, numBytes);

		if( !pFrame || !buffer )
			throwXAVException("error getting frame and/or buffer");
	}
	else if( pCodecCtx->codec_type == AVMEDIA_TYPE_AUDIO ) {
		;
	}

	nFramesDecoded = 0;
}

bool XAVStream::Decode(AVPacket *packet)
{
	if(pCodecCtx==NULL)
		return false;

	if( pCodecCtx->codec_type == AVMEDIA_TYPE_VIDEO ) {
		avcodec_decode_video2(pCodecCtx, pFrame, &frameFinished, packet);
		if( frameFinished ) {
			freeBuffs.Acquire();
			int size = pCodecCtx->height * pFrame->linesize[0];
			memcpy(buffer, pFrame->data[0], size);

			// for 4:2:0
			memcpy(buffer+size, pFrame->data[1], size/4);
			memcpy(buffer+size + (size/4), pFrame->data[2], size/4);
			usedBuffs.Release();
		}
	}
	else if( pCodecCtx->codec_type == AVMEDIA_TYPE_AUDIO ) {
	}

	return true;
}

unsigned char *XAVStream::GetBuffer() {
	usedBuffs.Acquire();
	return buffer;
}

void XAVStream::ReleaseBuffer() {
	freeBuffs.Release();
}

void XAVStream::Acquire() {
	freeBuffs.Acquire();
}

void XAVStream::Release() {
	usedBuffs.Release();
}

XAVSrc::XAVSrc(const std::string name) :
	name(name),
	mNumStreams(0),
	mVideoStream(NULL),
	mAudioStream(NULL),
	pFormatCtx(NULL)
{
	if(avformat_open_input(&pFormatCtx, name.c_str(), NULL, NULL)!=0) {
		throwXAVException("Couldn't open file: " + name);
	}

	mNumStreams = pFormatCtx->nb_streams;

	if( avformat_find_stream_info(pFormatCtx,NULL) < 0) {
		throwXAVException("Couldn't find stream info");
	}

	// for now, just grab the first video and first audio streams found.
	for(int i=0; i<mNumStreams; i++) {
		if ((pFormatCtx->streams[i]->codec->codec_type == AVMEDIA_TYPE_VIDEO) && !mVideoStream) {
			if(!mVideoStream) {
				mVideoStream = std::make_shared<XAVStream>(pFormatCtx->streams[i]->codec);
				mStreams.emplace_back(mVideoStream);
			}
		}
		else if ((pFormatCtx->streams[i]->codec->codec_type == AVMEDIA_TYPE_AUDIO) && !mAudioStream) {
			if (!mAudioStream) {
				mAudioStream = std::make_shared<XAVStream>(pFormatCtx->streams[i]->codec);
				mStreams.emplace_back(mAudioStream);
			}
		} 
		else
			mStreams.emplace_back(std::make_shared<XAVStream>(pFormatCtx->streams[i]->codec));
	}
}

XAVSrc::XAVSrc() :
	mNumStreams(0),
	mVideoStream(NULL),
	mAudioStream(NULL),
	pFormatCtx(NULL)
{
}

void *XAVSrc::Run()
{
	while( av_read_frame(pFormatCtx, &packet) >= 0 )
	{
		mStreams[packet.stream_index]->Decode(&packet);
		av_free_packet(&packet);
	}
	while(IsRunning())
	{
		mVideoStream->Acquire();
		mVideoStream->Release();
	}
	return NULL;
}

XAVStream *XAVSrc::VideoStream() {
	if(mVideoStream)
		return mVideoStream.get();
	else
		return NULL;
}

XAVStream *XAVSrc::AudioStream() {
	if(mAudioStream)
		return mAudioStream.get();
	else
		return NULL;
}
