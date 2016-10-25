#include "xgl.h"

XGLFramebuffer::XGLFramebuffer(int w, int h, GLuint *texs, int ntexs, bool d) :
	XGLObject("XGLFramebuffer"),
	width(w),
	height(h),
	numTextures(ntexs),
	hasDepth(d)
{
	if (ntexs > 8)
		throwXGLException("too many texures requested, max is 8, requested: " + std::to_string(ntexs));

	glGenFramebuffers(1, &fbo);
	GL_CHECK("glGenFramebuffers() failed");

	glBindFramebuffer(GL_FRAMEBUFFER, fbo);
	GL_CHECK("glBindFramebuffer() failed");

	for (int i = 0; i < numTextures; i++) {
		textures[i] = texs[i];
		attachments[i] = GL_COLOR_ATTACHMENT0 + i;

		glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0+i, GL_TEXTURE_2D, textures[i], 0);
		GL_CHECK("glFramebufferTexture() failedn");
	}

	glDrawBuffers(numTextures, attachments);
	GL_CHECK("glDrawBuffers() failed");

	if (hasDepth) {
		// The depth buffer
		glGenRenderbuffers(1, &depth);
		GL_CHECK("glGenRenderbuffers() failed");

		glBindRenderbuffer(GL_RENDERBUFFER, depth);
		GL_CHECK("glBindRenderbuffer() failed");

		glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT, width, height);
		GL_CHECK("glRenderbufferStorage() failed");

		glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, depth);
		GL_CHECK("glFramebufferRenderbuffer() failed");
	}

	if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
		xprintf("glCheckFramebufferStatus() != GL_FRAMEBUFFER_COMPLETE\n");

	glBindFramebuffer(GL_FRAMEBUFFER, 0);
	GL_CHECK("glBindFrameBuffer(0) failed");
}
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

XGLSharedFBO::XGLSharedFBO() : XSharedMem(DEFAULT_FILE_NAME) {
}

void XGLSharedFBO::Render(int width, int height) {
}