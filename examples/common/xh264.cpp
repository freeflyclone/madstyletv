#include "xh264.h"

extern "C" {
#include "h264Decoder.h"
#include "configfile.h"
#include "contributors.h"

#include "global.h"
#include "mbuffer.h"
#include "image.h"
#include "memalloc.h"
#include "sei.h"
#include "input.h"
#include "fast_memory.h"
}

namespace {
	InputParameters inputParameters{ 0 };

	// for decoded image display during development.  Ideally these
	// constants are decoded from the input MP4 stream, but we're
	// not there yet. So for now, just hard code them here, in one place.
	const int imageWidth = 1920;
	const int imageHeight = 1080;
	const int imageUVWidth = imageWidth / 2;
	const int imageUVHeight = imageHeight / 2;
	const int imageYSize = imageWidth * imageHeight;
	const int imageUVSize = imageYSize / 4;
	const int imageTotalSize = imageYSize + imageUVSize * 2;
}

// Hack: I changed output.c from the JM ldecod project to make it's internal
// "write_out_picture()" function a function pointer initialized to NULL, and
// in the one place it's called, a check for not-NULL is made before calling it.
// Thus, if we change this pointer to our own function, we get decoder output.
extern "C" { extern void(*write_out_picture)(VideoParameters*, StorablePicture*, int); };

void Xh264Decoder::_callback(VideoParameters* pvp, StorablePicture*psp, int p_out)
{
	Xh264Decoder* pDecoder = (Xh264Decoder*)pvp->p_Inp->p_ctx;

	if (pDecoder)
		pDecoder->InvokeCallbacks(pvp, psp, p_out);
}

void Xh264Decoder::InvokeCallbacks(VideoParameters* pvp, StorablePicture*psp, int p_out)
{
	for (auto fn : cbList)
		fn(pvp, psp, p_out);
}

void Xh264Decoder::AddCallback(Callback cb) {
	cbList.push_back(cb);
}

Xh264Decoder::Xh264Decoder() : XThread("Xh264Decoder"), XGLTexQuad()
{
	ParseCommand(&inputParameters, 0, nullptr);
	inputParameters.p_ctx = this;

	unsigned char *y = yuvBuffer;
	unsigned char *u = yuvBuffer + imageYSize;
	unsigned char *v = u + imageUVSize;

	AddTexture(imageWidth, imageHeight, 1, y);
	AddTexture(imageUVWidth, imageUVHeight, 1, u);
	AddTexture(imageUVWidth, imageUVHeight, 1, u);

	if (!write_out_picture)
		write_out_picture = _callback;
}

Xh264Decoder::~Xh264Decoder()
{
}

void Xh264Decoder::Draw()
{
	// assume the "yuv" shader was specified for this XGLShape,
	// hence we need 3 texture units for Y,U,V respectively.
	glProgramUniform1i(shader->programId, glGetUniformLocation(shader->programId, "texUnit0"), 0);
	glProgramUniform1i(shader->programId, glGetUniformLocation(shader->programId, "texUnit1"), 1);
	glProgramUniform1i(shader->programId, glGetUniformLocation(shader->programId, "texUnit2"), 2);

	unsigned char *y = yuvBuffer;
	unsigned char *u = yuvBuffer + imageYSize;
	unsigned char *v = u + imageUVSize;

	// Luma - Y
	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, texIds[0]);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RED, imageWidth, imageHeight, 0, GL_RED, GL_UNSIGNED_BYTE, (GLvoid *)y);
	GL_CHECK("glGetTexImage() didn't work");

	// Chroma - U
	glActiveTexture(GL_TEXTURE1);
	glBindTexture(GL_TEXTURE_2D, texIds[1]);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RED, imageUVWidth, imageUVHeight, 0, GL_RED, GL_UNSIGNED_BYTE, (GLvoid *)u);
	GL_CHECK("glGetTexImage() didn't work");

	// Chroma - V
	glActiveTexture(GL_TEXTURE2);
	glBindTexture(GL_TEXTURE_2D, texIds[2]);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RED, imageUVWidth, imageUVHeight, 0, GL_RED, GL_UNSIGNED_BYTE, (GLvoid *)v);
	GL_CHECK("glGetTexImage() didn't work");

	XGLTexQuad::Draw();
}


void Xh264Decoder::Run() {
	DecodedPicList *pDecPicList;

	OpenDecoder(&inputParameters);

	while (IsRunning())
		DecodeOneFrame(&pDecPicList);

	FinitDecoder(&pDecPicList);
	CloseDecoder();

	xprintf("Xh264Decoder::Run() finished\n");
}

