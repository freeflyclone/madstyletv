/**************************************************************
** R3DDevBuildScene.cpp
**
** Mission: Integrate R3DSDK fully, including CUDA
**************************************************************/
#include "ExampleXGL.h"
#include <cstdlib>

#include "R3DSDK.h"

using namespace R3DSDK;

class R3DPlayer : public XGLTexQuad {
public:
	R3DPlayer(const std::string& fname) : XGLTexQuad() {
		Clip* clip = InitializeClip(fname);

		m_width = clip->Width();
		m_height = clip->Height();

		Decode1stFrame(clip);

		GenR3DTextureBuffer(m_width, m_height);

		delete clip;
		FinalizeSdk();
	};

	~R3DPlayer() {
		if (unalignedImgbuffer)
			delete unalignedImgbuffer;
	}

	Clip* InitializeClip(const std::string& fname) {
		fileName = fname;
		const char* clipName = fileName.c_str();

		InitializeStatus initStat;

		initStat = InitializeSdk(".", OPTION_RED_NONE);
		if (initStat != ISInitializeOK)
		{
			xprintf("Failed to initialize SDK: %d\n", initStat);
			return nullptr;
		}

		// load the clip
		Clip *clip = new Clip(clipName);

		// let the user know if this failed
		if (clip->Status() != LSClipLoaded)
		{
			xprintf("Error loading %s: %d\n", clipName, clip->Status());
			delete clip;
			FinalizeSdk();
			return nullptr;
		}

		xprintf("Loaded %s\n", clipName);
		return clip;
	}

	uint16_t* AllocateAlignedHostBuffer(Clip* clip, size_t& memNeeded) {
		size_t width = clip->Width();
		size_t height = clip->Height();

		// three channels (RGB) in 16-bit (2 bytes) requires this much memory:
		memNeeded = width * height * 3 * 2;
		size_t adjusted = memNeeded + 16;

		// alloc this memory 16-byte aligned
		unalignedImgbuffer = new uint8_t[adjusted];
		if (unalignedImgbuffer == NULL) {
			xprintf("Failed to allocate %d bytes of memory for output image\n", static_cast<unsigned int>(memNeeded));
			return nullptr;
		}

		imgbuffer = (uint16_t*)(std::align(16, memNeeded, (void*&)unalignedImgbuffer, adjusted));
		return imgbuffer;
	}

	void Decode1stFrame(Clip* clip) {
		VideoDecodeJob job;

		size_t imgSize{ 0 };
		uint16_t* img = AllocateAlignedHostBuffer(clip, imgSize);

		// setup decoder parameters
		job.BytesPerRow = m_width * 2U;
		job.OutputBufferSize = imgSize;
		job.Mode = DECODE_FULL_RES_PREMIUM;
		job.OutputBuffer = img;
		job.PixelType = PixelType_16Bit_RGB_Planar;

		// decode the first frame (0) of the clip
		xprintf("Image is %d x %d\n", m_width, m_height);

		if (clip->DecodeVideoFrame(0U, job) != DSDecodeOK)
		{
			xprintf("Decode failed?\n");
			delete clip;
			FinalizeSdk();
			return;
		}
	}

	void GenR3DTextureBuffer(const int width, const int height) {
		// Output of R3D decoder is 16 bit planar, in RGB order
		// Each plane gets it's own texture unit, cuz that's super
		// easy, at the expense of being sub-optimal
		for (int i = 0; i < 3; i++) {
			GLuint texId;

			glGenTextures(1, &texId);
			glActiveTexture(GL_TEXTURE0 + numTextures);
			glBindTexture(GL_TEXTURE_2D, texId);
			glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
			glTexImage2D(GL_TEXTURE_2D, 0, GL_R16, width, height, 0, GL_RED, GL_UNSIGNED_SHORT, (void *)(imgbuffer + (width*height*i)));

			GL_CHECK("Eh, something failed");

			AddTexture(texId);
		}
	}

	void Draw()
	{
		if (imgbuffer == nullptr)
			return;

		// The "tex16planar" shader is require for R3DPlayer,
		// it uses a textureUnit per color, ie: R16, G16, and B16 respectively.
		glProgramUniform1i(shader->programId, glGetUniformLocation(shader->programId, "texUnit0"), 0);
		glProgramUniform1i(shader->programId, glGetUniformLocation(shader->programId, "texUnit1"), 1);
		glProgramUniform1i(shader->programId, glGetUniformLocation(shader->programId, "texUnit2"), 2);

		XGLTexQuad::Draw();
	}

private:
	int m_width{ 0 };
	int m_height{ 0 };
	uint8_t *unalignedImgbuffer{ nullptr };
	uint16_t *imgbuffer{ nullptr };
	std::string fileName;
};

void ExampleXGL::BuildScene() {
	XGLShape *shape;

	std::string r3DClipName = config.WideToBytes(config.Find(L"R3DFile")->AsString());

	AddShape("shaders/tex16planar", [&](){ shape = new R3DPlayer(r3DClipName); return shape; });
	glm::mat4 scale = glm::scale(glm::mat4(), glm::vec3(24.0f, 10.0f, 1.0f));
	glm::mat4 translate = glm::translate(glm::mat4(), glm::vec3(0, 0, 10.0f));
	glm::mat4 rotate = glm::rotate(glm::mat4(), glm::radians(90.0f), glm::vec3(1.0f, 0.0f, 0.0f));
	shape->model = translate * rotate * scale;
}
