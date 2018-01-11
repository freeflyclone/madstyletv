/**************************************************************
** MP4DevBuildScene.cpp
**
** Just to demonstrate instantiation of a "ground"
** plane and a single triangle, with default camera manipulation
** via keyboard and mouse.
**************************************************************/
#include "ExampleXGL.h"

#include "Ap4.h"
#include "xav.h"
#include "libavcodec/avcodec.h"

namespace {
	uint32_t spaceSize{ 2 };
}
class MyInspector : public AP4_AtomInspector {
public:
	MyInspector() {
		xprintf("MyInspector::MyInspector()\n");
		memset(spaces, 0, sizeof(spaces));
	}

	void StartAtom(const char* name, AP4_UI08 version, AP4_UI32 flags, AP4_Size header_size, AP4_UI64 size) {
		xprintf("%s%s, %d, %d\n", spaces, name, header_size, size);
		level++;
		memset(spaces, ' ', level * spaceSize);
		spaces[level * spaceSize] = 0;
	}

	void EndAtom() {
		if (level>0)
			level--;
		spaces[level * spaceSize] = 0;
	}

	void StartDescriptor(const char *name, AP4_Size headerSize, AP4_UI64 size) {
		xprintf("%s%s, %d, %d\n", spaces, name, headerSize, size);
	}

	void EndDescriptor() {}

	void AddField(const char *name, AP4_UI64 value, FormatHint hint = HINT_NONE) {
		xprintf("%s%s, %d\n", spaces, name, value);
	}
	void AddField(const char *name, float value, FormatHint hint = HINT_NONE) {
		xprintf("%s%s, %0.6f\n", spaces, name, value);
	}
	void AddField(const char *name, const char* value, FormatHint hint = HINT_NONE) {
		xprintf("%s%s, %s\n", spaces, name, value);
	}
	void AddField(const char *name, const unsigned char* value, AP4_Size byteCount, FormatHint hint = HINT_NONE) {
		xprintf("%s%s, %d\n", spaces, name, byteCount);
	}

	uint32_t level{ 0 };
	char spaces[512];
};

class MP4Demux {
public:
	MP4Demux(const char *filename) { 
		if (AP4_FAILED(result = AP4_FileByteStream::Create(filename, AP4_FileByteStream::STREAM_MODE_READ, input))) {
			xprintf("ERROR: cannot open input (%d)\n", result);
			return;
		}
		
		inspector = new MyInspector();
		inspector->SetVerbosity(0);
		while (atom_factory.CreateAtomFromStream(*input, atom) == AP4_SUCCESS) {
			// track stream position, in case some atom twiddles it.  
			AP4_Position position;
			input->Tell(position);

			atom->Inspect(*inspector);

			input->Seek(position);

			delete atom;
		}

		input->Seek(0);
		file = new AP4_File(*input);
		movie = file->GetMovie();
		videoTrack = movie->GetTrack(1);
		vCount = videoTrack->GetSampleCount();
		xprintf("Got %d samples\n",vCount);

		av_register_all();
		av_new_packet(&vPkt, 0x100000);
		vFrame = av_frame_alloc();

		if ((vCodec = avcodec_find_decoder(AV_CODEC_ID_H264)) != nullptr) {
			if ((vCtx = avcodec_alloc_context3(vCodec)) != nullptr) {
				if (avcodec_open2(vCtx, vCodec, nullptr) == 0) {
					xprintf("Successfully opened the video decoder\n");
				}
			}
		}
		

		for (AP4_Ordinal i = 0; i < vCount; i++) {
			videoTrack->GetSample(i, vSample);
			xprintf("sample %05d: %d @ %ld\n", i, vSample.GetSize(), vSample.GetOffset());
			vSample.ReadData(vBuffer);
			int dataSize = vBuffer.GetDataSize();
			vPkt.size = dataSize;
			memcpy(vPkt.data, vBuffer.GetData(), dataSize);
		}
	}

	AP4_Result result;
	AP4_ByteStream* input{ nullptr };
	AP4_Atom* atom;
	AP4_DefaultAtomFactory atom_factory;
	MyInspector* inspector;

	AP4_File *file;
	AP4_Movie *movie;
	AP4_Track *videoTrack,*audioTrack, *gpmfTracks[3];
	AP4_SampleDescription *vsd, *asd;
	AP4_Cardinal vCount;
	AP4_Sample vSample;
	AP4_DataBuffer vBuffer;

	AVCodec *vCodec;
	AVCodecContext *vCtx;
	AVPacket vPkt;
	AVFrame *vFrame;
};

void ExampleXGL::BuildScene() {
	preferredSwapInterval = 0;

	std::string videoUrl = config.WideToBytes(config.Find(L"VideoFile")->AsString());
	std::string videoPath;
	if (videoUrl.find("http") != videoUrl.npos)
		videoPath = videoUrl;
	else if (videoUrl.find(":", 1) != videoUrl.npos)
		videoPath = videoUrl;
	else
		videoPath = pathToAssets + "/" + videoUrl;

	MP4Demux mp4(videoPath.c_str());

	XGLShape *shape;
	AddShape("shaders/000-simple", [&](){ shape = new XGLTriangle(); return shape; });
}
