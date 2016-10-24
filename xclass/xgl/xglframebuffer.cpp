#include "xgl.h"

XGLFramebuffer::XGLFramebuffer(int w, int h, GLuint tex0, GLuint tex1, GLuint tex2) :
	XGLObject("XGLFramebuffer"),
	width(w),
	height(h)
{
	textures[0] = tex0;
	textures[1] = tex1;
	textures[2] = tex2;

	const int nSamples = 8;

	attachments[0] = GL_COLOR_ATTACHMENT0;
	attachments[1] = GL_COLOR_ATTACHMENT1;
	attachments[2] = GL_COLOR_ATTACHMENT2;

	xprintf("XGLFramebuffer::XGLFramebuffer()\n");

	glGenFramebuffers(1, &fbo);
	GL_CHECK("glGenFramebuffers() failed");

	glBindFramebuffer(GL_FRAMEBUFFER, fbo);
	GL_CHECK("glBindFramebuffer() failed");

	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, textures[0], 0);
	GL_CHECK("glFramebufferTexture() failedn");

	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT1, GL_TEXTURE_2D, textures[1], 0);
	GL_CHECK("glFramebufferTexture() failedn");

	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT2, GL_TEXTURE_2D, textures[2], 0);
	GL_CHECK("glFramebufferTexture() failedn");

	glDrawBuffers(3, attachments);
	GL_CHECK("glDrawBuffers() failed");

	// The depth buffer
	glGenRenderbuffers(1, &depth);
	GL_CHECK("glGenRenderbuffers() failed");

	glBindRenderbuffer(GL_RENDERBUFFER, depth);
	GL_CHECK("glBindRenderbuffer() failed");

	glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT, width, height);
	GL_CHECK("glRenderbufferStorage() failed");

	glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, depth);
	GL_CHECK("glFramebufferRenderbuffer() failed");

	if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
		xprintf("glCheckFramebufferStatus() != GL_FRAMEBUFFER_COMPLETE\n");

	glBindFramebuffer(GL_FRAMEBUFFER, 0);
	GL_CHECK("glBindFrameBuffer(0) failed");
}

XGLFramebuffer::XGLFramebuffer(int w, int h) :
	XGLObject("XGLFramebuffer"),
	width(w),
	height(h)
{
	const int nSamples = 8;

	attachments[0] = GL_COLOR_ATTACHMENT0;
	attachments[1] = GL_COLOR_ATTACHMENT1;

	xprintf("XGLFramebuffer::XGLFramebuffer()\n");

	glGenFramebuffers(1, &fbo);
	GL_CHECK("glGenFramebuffers() failed");

	glBindFramebuffer(GL_FRAMEBUFFER, fbo);
	GL_CHECK("glBindFramebuffer() failed");

	glGenTextures(2, textures);
	GL_CHECK("glGenTextures() failed");

	glBindTexture(GL_TEXTURE_2D, textures[0]);
	GL_CHECK("glBindTexture() failed");

	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, NULL);
	GL_CHECK("glTexImage2D() failed");

	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, textures[0], 0);
	GL_CHECK("glFramebufferTexture() failedn");

	glBindTexture(GL_TEXTURE_2D, textures[1]);
	GL_CHECK("glBindTexture() failed");

	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, NULL);
	GL_CHECK("glTexImage2D() failed");

	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT1, GL_TEXTURE_2D, textures[1], 0);
	GL_CHECK("glFramebufferTexture() failedn");

	glDrawBuffers(2, attachments);
	GL_CHECK("glDrawBuffers() failed");

	// The depth buffer
	glGenRenderbuffers(1, &depth);
	GL_CHECK("glGenRenderbuffers() failed");

	glBindRenderbuffer(GL_RENDERBUFFER, depth);
	GL_CHECK("glBindRenderbuffer() failed");

	glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT, width, height);
	GL_CHECK("glRenderbufferStorage() failed");

	glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, depth);
	GL_CHECK("glFramebufferRenderbuffer() failed");

	glBindFramebuffer(GL_FRAMEBUFFER, 0);
	GL_CHECK("glBindFrameBuffer(0) failed");

	glBindTexture(GL_TEXTURE_2D, 0);
	GL_CHECK("glBindTexture(0) failed");

	glBindRenderbuffer(GL_RENDERBUFFER, 0);
	GL_CHECK("glBindRenderbuffer() failed");

	if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
		xprintf("glCheckFramebufferStatus() != GL_FRAMEBUFFER_COMPLETE\n");
};

XGLFramebuffer::~XGLFramebuffer() {
	xprintf("XGLFramebuffer::~XGLFramebuffer()\n");
}

void XGLFramebuffer::Render(XGLFBORender renderFunc){
	glBindFramebuffer(GL_FRAMEBUFFER, fbo);
	GL_CHECK("glBindFramebuffer() failed");

	renderFunc();

	glBindFramebuffer(GL_FRAMEBUFFER, 0);
	GL_CHECK("glBidFramebuffer(0) failed");
}

void XGLSharedFBO::Render(XGLFBORender renderFunc) {
	width = pHeader->width;
	height = pHeader->height;

	XGLFramebuffer::Render(renderFunc);

	glBindFramebuffer(GL_READ_FRAMEBUFFER, fbo);
	GL_CHECK("glBindFrameBuffer(GL_READ_FRAMEBUFFER,fb->fbo) failed");

	glReadBuffer(GL_COLOR_ATTACHMENT0);
	GL_CHECK("glReadBuffer() failed");

	glReadPixels(0, 0, width, height, GL_BGR, GL_UNSIGNED_BYTE, mappedBuffer);
	GL_CHECK("glReadPixels() failed");

	glBindFramebuffer(GL_FRAMEBUFFER, 0);
}