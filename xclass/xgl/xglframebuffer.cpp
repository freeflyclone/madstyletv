#include "xgl.h"

XGLFramebuffer::XGLFramebuffer() : XGLObject("XGLFramebuffer") {
	xprintf("XGLFramebuffer::XGLFramebuffer()\n");

	glGenFramebuffers(1, &fbo);
	GL_CHECK("glGenFramebuffers() failed\n");

	glBindFramebuffer(GL_FRAMEBUFFER, fbo);
	GL_CHECK("glBindFramebuffer() failed\n");

	glGenTextures(1, &texture);
	GL_CHECK("glGenTextures() failed\n");

	glBindTexture(GL_TEXTURE_2D, texture);
	GL_CHECK("glBindTexture() failed\n");

	// Give an empty image to OpenGL ( the last "0" )
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, RENDER_WIDTH, RENDER_HEIGHT, 0, GL_RGB, GL_UNSIGNED_BYTE, 0);
	GL_CHECK("glTexImage2D() failed\n");

	// Poor filtering. Needed !
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	GL_CHECK("glTexParameter() failed\n");
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	GL_CHECK("glTexParameter() failed\n");

	// The depth buffer
	glGenRenderbuffers(1, &depthrenderbuffer);
	GL_CHECK("glGenRenderbuffers() failed\n");
	glBindRenderbuffer(GL_RENDERBUFFER, depthrenderbuffer);
	GL_CHECK("glBindRenderbuffer() failed\n");
	glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT, RENDER_WIDTH, RENDER_HEIGHT);
	GL_CHECK("glRenderbufferStorage() failed\n");
	glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, depthrenderbuffer);
	GL_CHECK("glFramebufferRenderbuffer() failed\n");

	// Set "renderedTexture" as our colour attachement #0
	glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, texture, 0);
	GL_CHECK("glFramebufferTexture() failed\n");

	// Set the list of draw buffers.
	GLenum DrawBuffers[1] = { GL_COLOR_ATTACHMENT0 };
	glDrawBuffers(1, DrawBuffers); // "1" is the size of DrawBuffers

	if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
		xprintf("glCheckFramebufferStatus() != GL_FRAMEBUFFER_COMPLETE\n");


	// file mapping stuff (Windows specific)
	{
		hFile = CreateFile(TEXT("C:\\vcam_buffer.dat"), (GENERIC_READ | GENERIC_WRITE), (FILE_SHARE_READ | FILE_SHARE_WRITE), NULL, OPEN_ALWAYS, FILE_ATTRIBUTE_NORMAL, NULL);
		if (hFile == INVALID_HANDLE_VALUE)
			OutputDebugString(TEXT("Failed to open file mapping file C:\\vcam_buffer.dat\n"));

		hMapping = CreateFileMapping(hFile, NULL, PAGE_READWRITE, 0, FILEMAPPING_SIZE, NULL);
		if (hMapping == NULL)
			OutputDebugString(TEXT("Failed to creat file mapping\n"));
		else {
			fileMappedBuffer = (unsigned char *)MapViewOfFile(hMapping, FILE_MAP_ALL_ACCESS, 0, 0, FILEMAPPING_SIZE);
			if (fileMappedBuffer == NULL) {
				OutputDebugString(TEXT("MapViewOfFile() failed\n"));
			}
		}
	}
};

XGLFramebuffer::~XGLFramebuffer() {
	xprintf("XGLFramebuffer::~XGLFramebuffer()\n");
}