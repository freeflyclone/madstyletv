#include "xgl.h"

XGLFramebuffer::XGLFramebuffer() : XGLObject("XGLFramebuffer") {
	const int nSamples = 8;
	xprintf("XGLFramebuffer::XGLFramebuffer()\n");

	glGenFramebuffers(1, &fbo);
	GL_CHECK("glGenFramebuffers() failed");

	glBindFramebuffer(GL_FRAMEBUFFER, fbo);
	GL_CHECK("glBindFramebuffer() failed");

	glGenTextures(1, &texture);
	GL_CHECK("glGenTextures() failed");

	glBindTexture(GL_TEXTURE_2D_MULTISAMPLE, texture);
	GL_CHECK("glBindTexture() failed");

	glTexImage2DMultisample(GL_TEXTURE_2D_MULTISAMPLE, nSamples, GL_RGB, RENDER_WIDTH, RENDER_HEIGHT, GL_TRUE);
	GL_CHECK("glTexImage2D() failed");

	glBindTexture(GL_TEXTURE_2D_MULTISAMPLE, 0);
	GL_CHECK("glBindTexture(0) failed");

	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D_MULTISAMPLE, texture, 0);
	GL_CHECK("glFramebufferTexture() failedn");

	// The depth buffer
	glGenRenderbuffers(1, &depth);
	GL_CHECK("glGenRenderbuffers() failed");

	glBindRenderbuffer(GL_RENDERBUFFER, depth);
	GL_CHECK("glBindRenderbuffer() failed");

	glRenderbufferStorageMultisample(GL_RENDERBUFFER, nSamples, GL_DEPTH_COMPONENT, RENDER_WIDTH, RENDER_HEIGHT);
	GL_CHECK("glRenderbufferStorage() failed");

	glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, depth);
	GL_CHECK("glFramebufferRenderbuffer() failed");

	glBindFramebuffer(GL_FRAMEBUFFER, 0);
	GL_CHECK("glBindFrameBuffer(0) failed");

	if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
		xprintf("glCheckFramebufferStatus() != GL_FRAMEBUFFER_COMPLETE\n");

	// now let's generate the intermediate FBO...
	glGenFramebuffers(1, &intFbo);
	GL_CHECK("glGenFramebuffers() failed");

	glBindFramebuffer(GL_FRAMEBUFFER, intFbo);
	GL_CHECK("glBindFramebuffer() failed");

	glGenTextures(1, &intTexture);
	GL_CHECK("glGenTextures() failed");

	glBindTexture(GL_TEXTURE_2D, intTexture);
	GL_CHECK("glBindTexture() failed");

	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, RENDER_WIDTH, RENDER_HEIGHT, 0, GL_RGB, GL_UNSIGNED_BYTE, 0);
	GL_CHECK("glTexImage2D() failed");

	glBindTexture(GL_TEXTURE_2D, 0);
	GL_CHECK("glBindTexture(0) failed");

	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, intTexture, 0);
	GL_CHECK("glFramebufferTexture() failedn");

	if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
		xprintf("glCheckFramebufferStatus() != GL_FRAMEBUFFER_COMPLETE\n");

	glBindFramebuffer(GL_FRAMEBUFFER, 0);
	GL_CHECK("glBindFrameBuffer(0) failed");
};

XGLFramebuffer::~XGLFramebuffer() {
	xprintf("XGLFramebuffer::~XGLFramebuffer()\n");
}