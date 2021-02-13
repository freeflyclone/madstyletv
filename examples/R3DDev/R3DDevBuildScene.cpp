/**************************************************************
** R3DDevBuildScene.cpp
**
** Mission: Integrate R3DSDK fully, including CUDA
**************************************************************/
#include "ExampleXGL.h"
#include <cstdlib>

#include "R3DSDK.h"

class R3DPlayer : public XGLTexQuad {
public:
	R3DPlayer(const std::string& fname) : fileName(fname), XGLTexQuad() {
		using namespace R3DSDK;
		const char* clipName = fileName.c_str();

		InitializeStatus initStat;

		initStat = InitializeSdk(".", OPTION_RED_NONE);
		if (initStat != ISInitializeOK)
		{
			xprintf("Failed to initialize SDK: %d\n", initStat);
			return;
		}

		// load the clip
		Clip *clip = new Clip(clipName);

		// let the user know if this failed
		if (clip->Status() != LSClipLoaded)
		{
			xprintf("Error loading %s: %d\n", clipName, clip->Status());
			delete clip;
			FinalizeSdk();
			return;
		}

		xprintf("Loaded %s\n", clipName);

		size_t width = clip->Width();
		size_t height = clip->Height();

		// three channels (RGB) in 16-bit (2 bytes) requires this much memory:
		size_t memNeeded = width * height * 3U * 2;
		size_t adjusted = memNeeded + 16;

		// alloc this memory 16-byte aligned
		unalignedImgbuffer = new uint8_t[adjusted];
		imgbuffer = (uint16_t*)(std::align(16, memNeeded, (void*&)unalignedImgbuffer, adjusted));

		if (imgbuffer == NULL) {
			xprintf("Failed to allocate %d bytes of memory for output image\n", static_cast<unsigned int>(memNeeded));
			return;
		}

		xprintf("Width: %d, Height: %d\n", width, height);

		// create and fill out a decode job structure so the
		// decoder knows what you want it to do
		VideoDecodeJob job;

		// setup decoder parameters
		job.BytesPerRow = width * 2U;
		job.OutputBufferSize = memNeeded;
		job.Mode = DECODE_FULL_RES_PREMIUM;
		job.OutputBuffer = imgbuffer;
		job.PixelType = PixelType_16Bit_RGB_Planar;

		// decode the first frame (0) of the clip
		xprintf("Decoding image at %d x %d\n", static_cast<unsigned int>(width), static_cast<unsigned int>(height));

		if (clip->DecodeVideoFrame(0U, job) != DSDecodeOK)
		{
			xprintf("Decode failed?\n");
			delete clip;
			FinalizeSdk();
			return;
		}

		GenR3DTextureBuffer(width, height);

		delete clip;
		FinalizeSdk();
	};

	void GenR3DTextureBuffer(const int width, const int height) {
		for (int i = 0; i < 3; i++) {
			GLuint texId;

			glGenTextures(1, &texId);
			GL_CHECK("glGetTextures() failed");

			glActiveTexture(GL_TEXTURE0 + numTextures);
			GL_CHECK("glActiveTexture(GL_TEXTURE0) failed");

			glBindTexture(GL_TEXTURE_2D, texId);
			GL_CHECK("glBindTexture() failed");

			glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
			GL_CHECK("glPixelStorei() failes");

			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
			GL_CHECK("glTexParameteri() failed");
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
			GL_CHECK("glTexParameteri() failed");
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
			GL_CHECK("glTexParameteri() failed");
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
			GL_CHECK("glTexParameteri() failed");

			glTexImage2D(GL_TEXTURE_2D, 0, GL_R16, width, height, 0, GL_RED, GL_UNSIGNED_SHORT, (void *)(imgbuffer + (width*height*i)));
			GL_CHECK("glTexParameteri() failed");

			AddTexture(texId);
		}
	}

	void Draw()
	{
		if (imgbuffer == nullptr)
			return;

		// assume the "tex16planar" shader was specified for this XGLShape,
		// hence we need 3 texture units for R16, G16, and B16 respectively.
		glProgramUniform1i(shader->programId, glGetUniformLocation(shader->programId, "texUnit0"), 0);
		glProgramUniform1i(shader->programId, glGetUniformLocation(shader->programId, "texUnit1"), 1);
		glProgramUniform1i(shader->programId, glGetUniformLocation(shader->programId, "texUnit2"), 2);

		XGLTexQuad::Draw();
	}

private:
	int width{ 0 };
	int height{ 0 };
	uint8_t *unalignedImgbuffer{ nullptr };
	uint16_t *imgbuffer{ nullptr };
	std::string fileName;
};

void ExampleXGL::BuildScene() {
	XGLShape *shape;

	std::string r3DClipName = config.WideToBytes(config.Find(L"R3DFile")->AsString());

	AddShape("shaders/tex16planar", [&](){ shape = new R3DPlayer(r3DClipName); return shape; });
	glm::mat4 scale = glm::scale(glm::mat4(), glm::vec3(24.0f, 10.0f, 1.0f));
	glm::mat4 translate = glm::translate(glm::mat4(), glm::vec3(0, 0, 5.4f));
	glm::mat4 rotate = glm::rotate(glm::mat4(), glm::radians(90.0f), glm::vec3(1.0f, 0.0f, 0.0f));
	shape->model = translate * rotate * scale;
}
