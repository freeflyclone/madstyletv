#include "xavsrc.h"

XAVStream::XAVStream(AVCodecContext *ctx) :	freeBuffs(XAV_NUM_FRAMES), streamIdx(0) {
	pCodecCtx = ctx;

	pCodec=avcodec_find_decoder(pCodecCtx->codec_id);
	if(pCodec==NULL)
		throwXAVException("codec not found");

	if( avcodec_open2(pCodecCtx, pCodec,NULL) < 0 )
		throwXAVException("Failed to open codec");

	pFrame = av_frame_alloc();
	if (!pFrame)
		throwXAVException("error allocating getting AVFrame\n");

	if (pCodecCtx->codec_type == AVMEDIA_TYPE_VIDEO) {
		xprintf("Found AVMEDIA_TYPE_VIDEO\n");
		xprintf("               width: %d\n", pCodecCtx->width);
		xprintf("              height: %d\n", pCodecCtx->height);
		numBytes = av_image_get_buffer_size(pCodecCtx->pix_fmt, pCodecCtx->width, pCodecCtx->height,8) * 4;

		AllocateBufferPool(XAV_NUM_FRAMES, numBytes, 1);
		width = pCodecCtx->width;
		height = pCodecCtx->height;
	}
	else if (pCodecCtx->codec_type == AVMEDIA_TYPE_AUDIO) {
		xprintf("Found AVMEDIA_TYPE_AUDIO\n");
		xprintf("  Samples Per Second: %d\n", pCodecCtx->sample_rate);
		xprintf("        SampleFormat: %d\n", pCodecCtx->sample_fmt);
		xprintf("          BlockAlign: %d\n", pCodecCtx->block_align);
		xprintf("            Channels: %d\n", pCodecCtx->channels);

		channels = pCodecCtx->channels;
		formatSize = 0;
		isFloat = false;
		sampleRate = pCodecCtx->sample_rate;

		switch (pCodecCtx->sample_fmt) {
			case 8:
				formatSize = 4;
				isFloat = true;
				break;

			default:
				throwXAVException("unknown pCodecCtx->sampleFmt");
		}
		// defer AllocateBufferPool() until first audio frame is decoded
	}
	else
		xprintf("Found unknown AVMEDIA_TYPE\n");

	nFramesDecoded = 0;
	nFramesRead = 0;
}

void XAVStream::AllocateBufferPool(int number, int size, int channels) {
	// TODO: make this number dynamic
	if (number != XAV_NUM_FRAMES)
		throwXAVException("number != XAV_NUM_FRAMES");

	if ((channels<1) || (channels>XAV_MAX_CHANNELS))
		throwXAVException("channels count is out of bounds: "+std::to_string(channels));

	for (int i = 0; i < number; i++) {
		unsigned char *buffer = (unsigned char *)av_malloc(size);
		memset(buffer, 0, size);
		frames[i].buffer = buffer;

		for (int j = 0; j < channels; j++) {
			unsigned char *buffer = (unsigned char *)av_malloc(size);
			frames[i].buffers[j] = buffer;
			memset(buffer, 0, size);
		}

		frames[i].nChannels = channels;
		frames[i].count = 0;
		frames[i].size = size;
	}
}

bool XAVStream::Decode(AVPacket *packet)
{
	if(pCodecCtx==NULL)
		return false;

	if( pCodecCtx->codec_type == AVMEDIA_TYPE_VIDEO ) {
		avcodec_decode_video2(pCodecCtx, pFrame, &frameFinished, packet);
		if( frameFinished ) {
			freeBuffs.wait_for(200);
			int frameIdx = (nFramesDecoded - 1) & (XAV_NUM_FRAMES - 1);
			XAVBuffer xb = frames[frameIdx];
			int size = pCodecCtx->height * pFrame->linesize[0];
			memcpy(xb.buffer, pFrame->data[0], size);

			// for 4:2:0
			memcpy(xb.buffer+size, pFrame->data[1], size/4);
			memcpy(xb.buffer+size + (size/4), pFrame->data[2], size/4);
			nFramesDecoded++;
			usedBuffs.notify();
		}
	}
	else if( pCodecCtx->codec_type == AVMEDIA_TYPE_AUDIO ) {
		int length = avcodec_decode_audio4(pCodecCtx, pFrame, &frameFinished, packet);
		if (frameFinished){
			if (nFramesDecoded == 0)
				AllocateBufferPool(XAV_NUM_FRAMES, pFrame->nb_samples * formatSize, channels);

			freeBuffs.wait_for(200);

			// replace all of this mumbo with an XFifo
			int frameIdx = (nFramesDecoded - 1) & (XAV_NUM_FRAMES - 1);
			XAVBuffer xb = frames[frameIdx];

			xb.nChannels = channels;

			for (int i = 0; i < channels; i++)
				memcpy(xb.buffers[i], pFrame->data[i], xb.size);

			nFramesDecoded++;
			usedBuffs.notify();
		}
	}

	return true;
}

XAVBuffer XAVStream::GetBuffer() {
	if (!usedBuffs.wait_for(200)) 
		return XAVBuffer {NULL, 0, 0};

	int frameIdx = nFramesRead & (XAV_NUM_FRAMES - 1);
	XAVBuffer xb = frames[frameIdx];
	freeBuffs.notify();
	nFramesRead++;
	return xb;
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

XAVSrc::XAVSrc(const std::string name, bool video=true, bool audio=true) :
	name(name),
	mNumStreams(0),
	mUsedStreams(0),
	mVideoStream(NULL),
	mAudioStream(NULL),
	pFormatCtx(NULL),
	doVideo(video),
	doAudio(audio),
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
		if (pFormatCtx->streams[i]->codec->codec_type == AVMEDIA_TYPE_VIDEO) {
			if(!mVideoStream) {
				if (doVideo) {
					mVideoStream = std::make_shared<XAVStream>(pFormatCtx->streams[i]->codec);
					mVideoStream->streamIdx = i;
					mStreams.emplace_back(mVideoStream);
					mUsedStreams++;
				}
			}
		}
		else if (pFormatCtx->streams[i]->codec->codec_type == AVMEDIA_TYPE_AUDIO) {
			if (!mAudioStream) {
				if (doAudio) {
					mAudioStream = std::make_shared<XAVStream>(pFormatCtx->streams[i]->codec);
					mAudioStream->streamIdx = i;
					mStreams.emplace_back(mAudioStream);
					mUsedStreams++;
				}
			}
		}
		else
			xprintf("Unknown codec type: %d, id: %d\n", pFormatCtx->streams[i]->codec->codec_type, pFormatCtx->streams[i]->codec);
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
		//if (packet.stream_index <= 1) {
		for (int i = 0; i < mUsedStreams; i++) {
			if (mStreams[i]->streamIdx == packet.stream_index) {
				mStreams[i]->Decode(&packet);
				av_packet_unref(&packet);
			}
		}
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
