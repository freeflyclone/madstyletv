/**************************************************************
** R3DDevBuildScene.cpp
**
** Mission: Integrate R3DSDK fully, including CUDA
**************************************************************/
#include "ExampleXGL.h"

#define __BASEFILE__ (strrchr(__FILE__, '\\') + 1)
#define FUNCENTER (xprintf("%s:%d: >> %s()\n", __BASEFILE__, __LINE__, __FUNCTION__))
#define FUNCEXIT  (xprintf("%s:%d: << %s()\n", __BASEFILE__, __LINE__, __FUNCTION__))
#define LOG(fmt, ...) { xprintf("%s:%d: " fmt "\n", __BASEFILE__, __LINE__, __VA_ARGS__); }

#include "XGLREDCuda.h"

class R3DPlayer : public XGLREDCuda, public XThread {
public:
	R3DPlayer(const std::string& fname) : XGLREDCuda(), XThread("R3DPlayerThread"), fileName(fname) {
		FUNCENTER;

		R3DSDK::Clip *clip = new R3DSDK::Clip(fileName.c_str());
		if (clip->Status() != R3DSDK::LSClipLoaded)
		{
			LOG("Failed to load clip %d", clip->Status());
			return;
		}

		m_width = clip->Width();
		m_height = clip->Height();

		GenR3DInterleavedTextureBuffer(m_width, m_height);

		LOG("Clip resolution = %u x %u", m_width, m_height);

		m_decodeJob = new R3DSDK::AsyncDecompressJob();

		m_decodeJob->Clip = clip;
		m_decodeJob->Mode = R3DSDK::DECODE_FULL_RES_PREMIUM;
		m_decodeJob->OutputBufferSize = R3DSDK::GpuDecoder::GetSizeBufferNeeded(*m_decodeJob);
		size_t adjustedSize = m_decodeJob->OutputBufferSize;
		m_decodeJob->OutputBuffer = AlignedMalloc(adjustedSize);
		m_decodeJob->VideoFrameNo = 200;
		m_decodeJob->VideoTrackNo = 0;
		m_decodeJob->Callback = CpuCallback;
		m_decodeJob->PrivateData = this;

		FUNCEXIT;
	};

	~R3DPlayer() {
	}

	static void CpuCallback(R3DSDK::AsyncDecompressJob * item, R3DSDK::DecodeStatus decodeStatus)
	{
		R3DPlayer* pThis = (R3DPlayer*)(item->PrivateData);
		if (pThis) {
			FUNCENTER;
			LOG("pThis->fileName: %s", pThis->fileName.c_str());

			pThis->JobQueue.push(item);

			FUNCEXIT;
		}
	}

	void Run() {
		FUNCENTER;

		if (m_pGpuDecoder->DecodeForGpuSdk(*m_decodeJob) != R3DSDK::DSDecodeOK)
		{
			printf("GPU decode submit failed\n");
			return;
		}

		while (IsRunning()) {
			std::this_thread::sleep_for(std::chrono::duration<int, std::milli>(100));
		}

		FUNCEXIT;
	}


private:
	std::string fileName;

	R3DSDK::AsyncDecompressJob* m_decodeJob{ nullptr };

	uint16_t* m_imgbuffer{ nullptr };
	uint8_t* m_unalignedImgbuffer{ nullptr };
	size_t m_imgSize{ 0 };
	size_t m_width, m_height;
};

R3DPlayer *player;

void ExampleXGL::BuildScene() {

	std::string r3DClipName = config.WideToBytes(config.Find(L"R3DFile")->AsString());

	AddShape("shaders/tex", [&](){ player = new R3DPlayer(r3DClipName); return player; });

	glm::mat4 scale = glm::scale(glm::mat4(), glm::vec3(16.0f, 9.0f, 1.0f));
	glm::mat4 translate = glm::translate(glm::mat4(), glm::vec3(0, 0, 9.0f));
	glm::mat4 rotate = glm::rotate(glm::mat4(), glm::radians(90.0f), glm::vec3(1.0f, 0.0f, 0.0f));
	player->model = translate * rotate * scale;

	player->SetAnimationFunction([&](float clock) {
		xprintf("%s\n", __FUNCTION__);
	});

	player->Start();
}
