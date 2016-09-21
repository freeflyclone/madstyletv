#include "xavsrc.h"

XAVStream::XAVStream(AVCodecContext *ctx) :	freeBuffs(XAV_NUM_FRAMES) {
	pCodecCtx = ctx;

	pCodec=avcodec_find_decoder(pCodecCtx->codec_id);
	if(pCodec==NULL)
		throwXAVException("codec not found");

	if( avcodec_open2(pCodecCtx, pCodec,NULL) < 0 )
		throwXAVException("Failed to open codec");

	if( pCodecCtx->codec_type == AVMEDIA_TYPE_VIDEO ) {
		xprintf("Found AVMEDIA_TYPE_VIDEO\n");
		xprintf("               width: %d\n", pCodecCtx->width);
		xprintf("              height: %d\n", pCodecCtx->height);
		pFrame = av_frame_alloc();
		numBytes = av_image_get_buffer_size(pCodecCtx->pix_fmt, pCodecCtx->width, pCodecCtx->height,8) * 4;
		buffer = (unsigned char *)av_malloc(numBytes);
		memset(buffer, 0, numBytes);

		for (int i = 0; i < XAV_NUM_FRAMES; i++) {
			frames[i].buffer = (unsigned char *)av_malloc(numBytes);
			frames[i].count = numBytes;
			frames[i].size = 0;
			memset(frames[i].buffer, 0, numBytes);
		}

		if (!pFrame || !buffer)
			throwXAVException("error getting frame and/or buffer\n");
	}
	else if (pCodecCtx->codec_type == AVMEDIA_TYPE_AUDIO) {
		char buff[1024];
		xprintf("Found AVMEDIA_TYPE_AUDIO\n");
		xprintf("  Samples Per Second: %d\n", pCodecCtx->sample_rate);
		xprintf("        SampleFormat: %s\n", av_get_sample_fmt_string(buff, sizeof(buff), pCodecCtx->sample_fmt));
		xprintf("          BlockAlign: %d\n", pCodecCtx->block_align);
		xprintf("            Channels: %d\n", pCodecCtx->channels);
		pFrame = av_frame_alloc();
		buffer = (unsigned char *)av_malloc(192000);

		for (int i = 0; i < XAV_NUM_FRAMES; i++) {
			frames[i].buffer = (unsigned char *)av_malloc(4096);
			frames[i].count = 4096;
			frames[i].size = 0;
			memset(frames[i].buffer, 0, 4096);
		}
	}
	else
		xprintf("Found unknown AVMEDIA_TYPE\n");

	nFramesDecoded = 0;
}

bool XAVStream::Decode(AVPacket *packet)
{
	if(pCodecCtx==NULL)
		return false;

	if( pCodecCtx->codec_type == AVMEDIA_TYPE_VIDEO ) {
		avcodec_decode_video2(pCodecCtx, pFrame, &frameFinished, packet);
		if( frameFinished ) {
			freeBuffs.wait_for(200);
			int size = pCodecCtx->height * pFrame->linesize[0];
			memcpy(buffer, pFrame->data[0], size);

			// for 4:2:0
			memcpy(buffer+size, pFrame->data[1], size/4);
			memcpy(buffer+size + (size/4), pFrame->data[2], size/4);
			nFramesDecoded++;
			usedBuffs.notify();
		}
	}
	else if( pCodecCtx->codec_type == AVMEDIA_TYPE_AUDIO ) {
		int length = avcodec_decode_audio4(pCodecCtx, pFrame, &frameFinished, packet);
		if (frameFinished){
			//xprintf("Decoded audio frame: found %d bytes.\n", length);
		}
	}

	return true;
}

unsigned char *XAVStream::GetBuffer() {
	usedBuffs.wait_for(200);
	freeBuffs.notify();
	return buffer;
}

void XAVStream::ReleaseBuffer() {
	freeBuffs.notify();
}

void XAVStream::Acquire() {
	freeBuffs.wait_for(200);
}

void XAVStream::Release() {
	usedBuffs.notify();
}

XAVSrc::XAVSrc(const std::string name) :
	name(name),
	mNumStreams(0),
	mVideoStream(NULL),
	mAudioStream(NULL),
	pFormatCtx(NULL),
	XThread(name)
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
	pFormatCtx(NULL),
	XThread("XAVSrc")
{
}

void XAVSrc::Run()
{
	while( av_read_frame(pFormatCtx, &packet) >= 0 )
	{
		if (packet.stream_index <= 1) {
			mStreams[packet.stream_index]->Decode(&packet);
			av_packet_unref(&packet);
		}
	}
	while(IsRunning())
	{
		mVideoStream->Acquire();
		mVideoStream->Release();
	}
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
