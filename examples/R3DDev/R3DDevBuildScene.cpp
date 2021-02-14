/**************************************************************
** R3DDevBuildScene.cpp
**
** Mission: Integrate R3DSDK fully, including CUDA
**************************************************************/
#include "ExampleXGL.h"
#include <cstdlib>
#include <cuda_runtime.h>
#include <string.h>
#include <stdio.h>

#define __BASEFILE__ (strrchr(__FILE__, '\\') + 1)
#define FUNCENTER (xprintf("%s:%d: >> %s()\n", __BASEFILE__, __LINE__, __FUNCTION__))
#define FUNCEXIT  (xprintf("%s:%d: << %s()\n", __BASEFILE__, __LINE__, __FUNCTION__))
#define LOG(fmt, ...) { xprintf("%s:%d: " fmt "\n", __BASEFILE__, __LINE__, __VA_ARGS__); }

#include "R3DSDK.h"
#include "R3DSDKCuda.h"
#include <R3DSDKDefinitions.h>

#include "XGLMemoryPool.h"
#include "XGLREDCuda.h"

class R3DPlayer : public XGLTexQuad, public XThread {
public:
	R3DPlayer(const std::string& fname) : XGLTexQuad(), XThread("R3DPlayerThread"), fileName(fname) {
		FUNCENTER;

		m_pXglRedCuda = new XGLREDCuda();

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

			pThis->m_pXglRedCuda->JobQueue.push(item);

			FUNCEXIT;
		}
	}

	void Run() {
		FUNCENTER;

		R3DSDK::Clip *clip = new R3DSDK::Clip(fileName.c_str());
		if (clip->Status() != R3DSDK::LSClipLoaded)
		{
			LOG("Failed to load clip %d", clip->Status());
			return;
		}

		LOG("Clip resolution = %u x %u", clip->Width(), clip->Height());

		R3DSDK::AsyncDecompressJob *job = new R3DSDK::AsyncDecompressJob();
		job->Clip = clip;
		job->Mode = R3DSDK::DECODE_FULL_RES_PREMIUM;
		job->OutputBufferSize = R3DSDK::GpuDecoder::GetSizeBufferNeeded(*job);
		size_t adjustedSize = job->OutputBufferSize;
		job->OutputBuffer = m_pXglRedCuda->AlignedMalloc(adjustedSize);
		job->VideoFrameNo = 0;
		job->VideoTrackNo = 0;
		job->Callback = CpuCallback;
		job->PrivateData = this;

		if (m_pXglRedCuda->m_pGpuDecoder->DecodeForGpuSdk(*job) != R3DSDK::DSDecodeOK)
		{
			printf("GPU decode submit failed\n");
			return;
		}

		while (IsRunning()) {
			LOG("%s", "running...");
			std::this_thread::sleep_for(std::chrono::duration<int, std::milli>(100));
		}

		FUNCEXIT;
	}

	void GenR3DInterleavedTextureBuffer(const int width, const int height) {
		GLuint texId;

		glGenTextures(1, &texId);
		glActiveTexture(GL_TEXTURE0 + numTextures);
		glBindTexture(GL_TEXTURE_2D, texId);
		glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB16, width, height, 0, GL_RGB, GL_UNSIGNED_SHORT, (void *)(m_imgbuffer));

		GL_CHECK("Eh, something failed");

		AddTexture(texId);
	}

private:
	std::string fileName;

	XGLREDCuda *m_pXglRedCuda;

	uint16_t* m_imgbuffer;
};

R3DPlayer *player;

void ExampleXGL::BuildScene() {

	std::string r3DClipName = config.WideToBytes(config.Find(L"R3DFile")->AsString());

	AddShape("shaders/tex", [&](){ player = new R3DPlayer(r3DClipName); return player; });

	glm::mat4 scale = glm::scale(glm::mat4(), glm::vec3(16.0f, 9.0f, 1.0f));
	glm::mat4 translate = glm::translate(glm::mat4(), glm::vec3(0, 0, 9.0f));
	glm::mat4 rotate = glm::rotate(glm::mat4(), glm::radians(90.0f), glm::vec3(1.0f, 0.0f, 0.0f));
	player->model = translate * rotate * scale;

	player->Start();
}
