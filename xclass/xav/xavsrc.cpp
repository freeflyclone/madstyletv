#include "xavsrc.h"

XAVStream::XAVStream(AVCodecContext *ctx) : 
	freeBuffs(numFrames), 
	pStream(NULL), 
	streamIdx(0), 
	streamTime(0.0), 
	totalChunks(0),
	totalBytes(0) 
{
	pCodecCtx = ctx;

	if (pCodecCtx) {
		pCodec = avcodec_find_decoder(pCodecCtx->codec_id);
		if (pCodec) {
			if (avcodec_open2(pCodecCtx, pCodec, NULL) < 0)
				throwXAVException("Failed to open codec");

			pFrame = av_frame_alloc();
			if (!pFrame)
				throwXAVException("error allocating getting AVFrame\n");

			if (pCodecCtx->codec_type == AVMEDIA_TYPE_VIDEO) {
				xprintf("Found AVMEDIA_TYPE_VIDEO\n");
				xprintf("               width: %d\n", pCodecCtx->width);
				xprintf("              height: %d\n", pCodecCtx->height);
				numBytes = av_image_get_buffer_size(pCodecCtx->pix_fmt, pCodecCtx->width, pCodecCtx->height, 8);

				AllocateBufferPool(numFrames, numBytes, 3);
				width = pCodecCtx->width;
				height = pCodecCtx->height;

				// figure out the chroma sub-sampling of the pixel format
				int pixelFormat = pCodecCtx->pix_fmt;
				const AVPixFmtDescriptor *pixDesc = av_pix_fmt_desc_get((AVPixelFormat)pixelFormat);

				// save it so our clients can know
				chromaWidth = pCodecCtx->width / (1 << pixDesc->log2_chroma_w);
				chromaHeight = pCodecCtx->height / (1 << pixDesc->log2_chroma_h);
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
	}
	else {
		AllocateBufferPool(numFrames, 0x8000, 1);
	}
}

void XAVStream::AllocateBufferPool(int number, int size, int channels) {
	// TODO: make this number dynamic
	if (number != numFrames)
		throwXAVException("number != numFrames");

	if ((channels<1) || (channels>maxChannels))
		throwXAVException("channels count is out of bounds: "+std::to_string(channels));

	for (int i = 0; i < number; i++) {
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
	if (pCodecCtx) {
		if (pCodecCtx->codec_type == AVMEDIA_TYPE_VIDEO) {
			avcodec_decode_video2(pCodecCtx, pFrame, &frameFinished, packet);
			if (frameFinished) {
				freeBuffs.wait_for(200);
				int frameIdx = (nFramesDecoded - 1) & (numFrames - 1);

				XAVBuffer xb = frames[frameIdx];

				// if linesize[x] == width, we can copy the frame with a single memcpy()
				if (pFrame->linesize[0] == width) {
					// luminance Y
					int ySize = height * pFrame->linesize[0];
					memcpy(xb.buffers[0], pFrame->data[0], ySize);

					// chrominance U
					int uSize = chromaWidth * chromaHeight;
					memcpy(xb.buffers[1], pFrame->data[1], uSize);

					// chrominance V
					int vSize = chromaWidth * chromaHeight;
					memcpy(xb.buffers[2], pFrame->data[2], uSize);
				}
				// ... otherwise it has to be scanline at a time.
				else {
					// luminance Y
					unsigned char *s = pFrame->data[0];
					unsigned char *d = xb.buffers[0];
					for (int i = 0; i < height; i++) {
						memcpy(d, s, pFrame->linesize[0]);
						s += pFrame->linesize[0];
						d += width;
					}

					// chrominance U
					s = pFrame->data[1];
					d = xb.buffers[1];
					for (int i = 0; i < chromaHeight; i++) {
						memcpy(d, s, pFrame->linesize[1]);
						s += pFrame->linesize[1];
						d += chromaWidth;
					}

					// chrominance V
					s = pFrame->data[2];
					d = xb.buffers[2];
					for (int i = 0; i < chromaHeight; i++) {
						memcpy(d, s, pFrame->linesize[2]);
						s += pFrame->linesize[2];
						d += chromaWidth;
					}
				}

				nFramesDecoded++;
				usedBuffs.notify();
			}
		}
		else if (pCodecCtx->codec_type == AVMEDIA_TYPE_AUDIO) {
			int length = avcodec_decode_audio4(pCodecCtx, pFrame, &frameFinished, packet);
			if (frameFinished){
				if (nFramesDecoded == 0)
					AllocateBufferPool(numFrames, pFrame->nb_samples * formatSize, channels);

				freeBuffs.wait_for(200);

				// replace all of this mumbo with an XFifo
				int frameIdx = (nFramesDecoded - 1) & (numFrames - 1);
				XAVBuffer xb = frames[frameIdx];

				xb.nChannels = channels;

				for (int i = 0; i < channels; i++)
					memcpy(xb.buffers[i], pFrame->data[i], xb.size);

				streamTime += (double)pFrame->nb_samples / (double)pFrame->sample_rate;
				nFramesDecoded++;
				usedBuffs.notify();
			}
		}
	}
	else {
		auto p = packet;
		InvokeDataFunctions(p->data, p->size, p->pts);
	}

	return true;
}

XAVStream::XAVBuffer XAVStream::GetBuffer() {
	if (!usedBuffs.wait_for(1000)) {
		xprintf("XAVStream::GetBuffer() timed out\n");
		return XAVBuffer{ { NULL, NULL, NULL }, 0, 0 };
	}

	int frameIdx = nFramesRead & (numFrames - 1);
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
					mVideoStream->pStream = pFormatCtx->streams[i];
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
					mAudioStream->pStream = pFormatCtx->streams[i];
					mAudioStream->streamIdx = i;
					mStreams.emplace_back(mAudioStream);
					mUsedStreams++;
				}
			}
		}
		else {
			std::shared_ptr<XAVStream> meta = std::make_shared<XAVStream>(nullptr);

			meta->pStream = pFormatCtx->streams[i];
			meta->streamIdx = i;
			meta->AddDataFunction([&,meta](uint8_t *b, size_t l, uint64_t t){
				if (meta->totalBytes > (meta->totalChunks * 0x8000)) {
					xprintf("Stream lambda: %d, %ld\n", meta->streamIdx, meta->totalBytes);
					meta->totalChunks++;
				}
			});

			mStreams.emplace_back(meta);
			mUsedStreams++;
			xprintf("Registered data stream id: %X\n", pFormatCtx->streams[i]->id);
		}
	}
	xprintf("XAVSrc::XAVSrc() complete, %s video, %s audio\n", (mVideoStream == NULL) ? "does not have" : "has", (mAudioStream == NULL) ? "does not have" : "has");
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
