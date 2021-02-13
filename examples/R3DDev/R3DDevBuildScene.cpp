/**************************************************************
** R3DDevBuildScene.cpp
**
** Mission: Integrate R3DSDK fully, including CUDA
**************************************************************/
#include "ExampleXGL.h"
#include <cstdlib>

#include "R3DSDK.h"
static const char* clipName = "BMX.RDC/A006_A050_0808WW_001.R3D";

class R3DPlayer : public XGLTexQuad {
public:
	R3DPlayer(std::string clipName, int width, int height) : width(width), height(height), XGLTexQuad() {
		GLuint texId;
		glGenTextures(1, &texId);
		GL_CHECK("glGetTextures() failed");

		glActiveTexture(GL_TEXTURE0 + numTextures);
		GL_CHECK("glActiveTexture(GL_TEXTURE0) failed");

		glBindTexture(GL_TEXTURE_2D, texId);
		GL_CHECK("glBindTexture() failed");

		glActiveTexture(GL_TEXTURE0 + numTextures);
		GL_CHECK("glActiveTexture(GL_TEXTURE0) failed");

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

		//glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_SHORT, img);
	};

	int width, height;
};

void ExampleXGL::BuildScene() {
	using namespace R3DSDK;

	XGLShape *shape;

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
	size_t memNeeded = width * height * 3U * 2U;
	size_t adjusted = memNeeded + 16;

	// alloc this memory 16-byte aligned
	unsigned char *unAlignedImgBuffer = new unsigned char[adjusted];
	unsigned char *imgbuffer = (unsigned char*)(std::align(16, memNeeded, (void*&)unAlignedImgBuffer, adjusted));

	if (imgbuffer == NULL)
	{
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

	delete clip;
	FinalizeSdk();

	AddShape("shaders/tex", [&](){ shape = new XGLTexQuad(width,height,4,imgbuffer); return shape; });
	delete unAlignedImgBuffer;

	glm::mat4 scale = glm::scale(glm::mat4(), glm::vec3(9.6f, 5.4f, 1.0f));
	glm::mat4 translate = glm::translate(glm::mat4(), glm::vec3(0, 0, 5.4f));
	glm::mat4 rotate = glm::rotate(glm::mat4(), glm::radians(90.0f), glm::vec3(1.0f, 0.0f, 0.0f));
	shape->model = translate * rotate * scale;
}
