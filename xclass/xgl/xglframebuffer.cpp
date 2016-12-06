#include "xgl.h"

XGLFramebuffer::XGLFramebuffer(int w, int h, bool c, bool d, GLuint t) : width(w), height(h), hasColor(c), hasDepth(d), numTextures(0) {
	glGenFramebuffers(1, &fbo);
	GL_CHECK("glGenFramebuffers() failed");

	glBindFramebuffer(GL_FRAMEBUFFER, fbo);
	GL_CHECK("glBindFramebuffer() failed");

	if (hasColor)
		AddColorAttachment(t);

	if (hasDepth)
		AddDepthBuffer();

	if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
		xprintf("glCheckFramebufferStatus() != GL_FRAMEBUFFER_COMPLETE\n");

	glBindFramebuffer(GL_FRAMEBUFFER, 0);
	GL_CHECK("glBindFrameBuffer(0) failed");
}

XGLFramebuffer::~XGLFramebuffer() {
	xprintf("XGLFramebuffer::~XGLFramebuffer()\n");
	// this should release any OpenGL resources that 
}

void XGLFramebuffer::AddColorAttachment(GLuint t, GLenum target, GLint format, GLenum internalFormat) {
	GLuint texId;

	if (numTextures==8)
		throwXGLException("FBO has enough color attachments already");

	glBindFramebuffer(GL_FRAMEBUFFER, fbo);
	GL_CHECK("glBindFramebuffer() failed");

	// caller didn't specify an existing texture buffer, so create one.
	if (t == 0) {
		glGenTextures(1, &texId);
		GL_CHECK("glGenTextures() failed");

		glBindTexture(GL_TEXTURE_2D, texId);
		GL_CHECK("glBindTexture() failed");

		glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
		GL_CHECK("glPixelStorei() failed");

		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
		GL_CHECK("glTexParameteri() failed");
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
		GL_CHECK("glTexParameteri() failed");
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
		GL_CHECK("glTexParameteri() failed");
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
		GL_CHECK("glTexParameteri() failed");

		glPixelStorei(GL_PACK_ALIGNMENT, 1);
		GL_CHECK("glPixelStorei() failed");

		glTexImage2D(GL_TEXTURE_2D, 0, internalFormat, width, height, 0, format, GL_UNSIGNED_BYTE, 0);
		GL_CHECK("glTexImage2D() failed");
	}
	else
		texId = t;

	textures[numTextures] = texId;
	attachments[numTextures] = GL_COLOR_ATTACHMENT0 + numTextures;

	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0 + numTextures, target, texId, 0);
	GL_CHECK("glFramebufferTexture() failedn");

	numTextures++;

	glDrawBuffers(numTextures, attachments);
	GL_CHECK("glDrawBuffers() failed");

	glBindFramebuffer(GL_FRAMEBUFFER, fbo);
	GL_CHECK("glBindFramebuffer() failed");

	// leave the texId bound, in case subsequent changes are desired.  (for example various glPixelStore() parameters)
}

void XGLFramebuffer::AddDepthBuffer() {
	glBindFramebuffer(GL_FRAMEBUFFER, fbo);
	GL_CHECK("glBindFramebuffer() failed");

	glGenRenderbuffers(1, &depth);
	GL_CHECK("glGenRenderbuffers() failed");

	glBindRenderbuffer(GL_RENDERBUFFER, depth);
	GL_CHECK("glBindRenderbuffer() failed");

	glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT, width, height);
	GL_CHECK("glRenderbufferStorage() failed");

	glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, depth);
	GL_CHECK("glFramebufferRenderbuffer() failed");

	glBindFramebuffer(GL_FRAMEBUFFER, fbo);
	GL_CHECK("glBindFramebuffer() failed");
}

void XGLFramebuffer::Render(XGLFBORender renderFunc){
	glBindFramebuffer(GL_FRAMEBUFFER, fbo);
	GL_CHECK("glBindFramebuffer() failed");

	renderFunc();

	glBindFramebuffer(GL_FRAMEBUFFER, 0);
	GL_CHECK("glBidFramebuffer(0) failed");
}

XGLSharedFBO::XGLSharedFBO(XGL *context) : XSharedMem(shmemDefaultFile), pXGL(context), msFbo(NULL), ssFbo(NULL), encoder(NULL), encWidth(0), encHeight(0) {
	// blit only FBO, no depth needed, adding multisampled color attachment
	msFbo = new XGLFramebuffer(XGLFramebuffer::renderWidth, XGLFramebuffer::renderHeight, false, false);

	// create a multi-sampled color buffer for "msFbo"
	glGenTextures(1, &texture);
	GL_CHECK("glGenTextures() failed");
	glBindTexture(GL_TEXTURE_2D_MULTISAMPLE, texture);
	GL_CHECK("glBindTexture() failed");
	glTexImage2DMultisample(GL_TEXTURE_2D_MULTISAMPLE, XGLFramebuffer::multiSamples, GL_RGB, XGLFramebuffer::renderWidth, XGLFramebuffer::renderHeight, GL_TRUE);
	GL_CHECK("glTexImage2D() failed");
	glBindTexture(GL_TEXTURE_2D_MULTISAMPLE, 0);
	GL_CHECK("glBindTexture(0) failed");

	msFbo->AddColorAttachment(texture, GL_TEXTURE_2D_MULTISAMPLE);

	ssFbo = new XGLFramebuffer(XGLFramebuffer::renderWidth, XGLFramebuffer::renderHeight, true, false);
	scaleSharedFbo = new XGLFramebuffer(XGLFramebuffer::renderWidth, XGLFramebuffer::renderHeight, true, false);
	scaleEncoderFbo = new XGLFramebuffer(XGLFramebuffer::renderWidth, XGLFramebuffer::renderHeight, true, false);
	sharedFbo = new XGLFramebuffer(XGLFramebuffer::renderWidth, XGLFramebuffer::renderHeight, true, false);
	encoderFbo = new XGLFramebuffer(XGLFramebuffer::renderWidth, XGLFramebuffer::renderHeight, true, false);

	// add additional color atachments for RGB -> YUV planar for XAVEncoder
	encoderFbo->AddColorAttachment(0, GL_TEXTURE_2D, GL_RED, GL_R8);
	encoderFbo->AddColorAttachment(0, GL_TEXTURE_2D, GL_RED, GL_R8);
	encoderFbo->AddColorAttachment(0, GL_TEXTURE_2D, GL_RED, GL_R8);

	MakeFlipQuad();

	if (pXGL->config.Find(L"Encoder.enabled")->AsBool()) {
		if ((encoder = new XAVEncoder(&(pXGL->config), yBuffer, uBuffer, vBuffer)) != NULL) {
			encWidth = encoder->ctx->width;
			encHeight = encoder->ctx->height;
			xprintf("Encoder is set to %d by %d\n", encWidth, encHeight);
		}
	}
}

void XGLSharedFBO::MakeFlipQuad() {
	pXGL->CreateShape("shaders/imageflip", [&]() { flipQuad = new XGLTexQuad(); return flipQuad; });
	flipQuad->AddTexture(scaleEncoderFbo->textures[0]);
	flipQuad->AddTexture(scaleSharedFbo->textures[0]);
}

void XGLSharedFBO::RenderFlipQuadToShared() {
	// render it to the "sharedFbo" for the DirectShow Vcam source filter
	glBindFramebuffer(GL_FRAMEBUFFER, sharedFbo->fbo);
	glViewport(0, 0, XGLFramebuffer::renderWidth, XGLFramebuffer::renderHeight);

	flipQuad->model = glm::mat4(1);
	flipQuad->Render(0.0);
}

void XGLSharedFBO::RenderFlipQuadToEncoder() {
	// render it again to the "encoderFbo" for RGB -> YUV conversion
	glBindFramebuffer(GL_FRAMEBUFFER, encoderFbo->fbo);
	glViewport(0, 0, XGLFramebuffer::renderWidth, XGLFramebuffer::renderHeight);

	// calculate how far to vertically offset the flipQuad so it's rendered correctly
	// in the encoder buffer(s). This amounts to a 2D translation in screen space
	int heightDiff = XGLFramebuffer::renderHeight - encHeight;
	float yOffset = (2 * ((float)heightDiff / (float)XGLFramebuffer::renderHeight));

	glm::mat4 scale = glm::scale(glm::mat4(), glm::vec3(1, -1, 1));
	glm::mat4 translate = glm::translate(glm::mat4(), glm::vec3(0, yOffset, 0));

	flipQuad->model = scale * translate;

	flipQuad->Render(0.0);

	glViewport(0, 0, vpWidth, vpHeight);
}

void XGLSharedFBO::RenderFlipQuad() {
	glProgramUniform1i(flipQuad->shader->programId, glGetUniformLocation(flipQuad->shader->programId, "texUnit0"), 0);
	glProgramUniform1i(flipQuad->shader->programId, glGetUniformLocation(flipQuad->shader->programId, "texUnit1"), 1);

	RenderFlipQuadToShared();
	RenderFlipQuadToEncoder();
}

void XGLSharedFBO::CopyScreenToFBO(){
	glBindFramebuffer(GL_READ_FRAMEBUFFER, 0);
	GL_CHECK("glBindFrameBuffer(0) failed");

	glBindFramebuffer(GL_DRAW_FRAMEBUFFER, msFbo->fbo);
	GL_CHECK("glBindFrameBuffer(DRAW) failed");

	glBlitFramebuffer(0, 0, vpWidth, vpHeight, 0, 0, vpWidth, vpHeight, GL_COLOR_BUFFER_BIT, GL_LINEAR);
	GL_CHECK("glBlitFramebuffer() failed");
}

void XGLSharedFBO::ResolveMultisampledFBO(){
	glBindFramebuffer(GL_READ_FRAMEBUFFER, msFbo->fbo);
	GL_CHECK("glBindFrameBuffer(0) failed");

	glBindFramebuffer(GL_DRAW_FRAMEBUFFER, ssFbo->fbo);
	GL_CHECK("glBindFrameBuffer(DRAW) failed");

	glBlitFramebuffer(0, 0, vpWidth, vpHeight, 0, 0, vpWidth, vpHeight, GL_COLOR_BUFFER_BIT, GL_LINEAR);
	GL_CHECK("glBlitFramebuffer() failed");
}

void XGLSharedFBO::ScaleToOutputSize(){
	glBindFramebuffer(GL_READ_FRAMEBUFFER, ssFbo->fbo);
	GL_CHECK("glBindFrameBuffer(0) failed");

	glBindFramebuffer(GL_DRAW_FRAMEBUFFER, scaleSharedFbo->fbo);
	GL_CHECK("glBindFrameBuffer(DRAW) failed");

	glBlitFramebuffer(0, 0, vpWidth, vpHeight, 0, 0, pHeader->width, pHeader->height, GL_COLOR_BUFFER_BIT, GL_LINEAR);
	GL_CHECK("glBlitFramebuffer() failed");

	glBindFramebuffer(GL_DRAW_FRAMEBUFFER, scaleEncoderFbo->fbo);
	GL_CHECK("glBindFrameBuffer(DRAW) failed");

	glBlitFramebuffer(0, 0, vpWidth, vpHeight, 0, 0, encWidth, encHeight, GL_COLOR_BUFFER_BIT, GL_LINEAR);
	GL_CHECK("glBlitFramebuffer() failed");
}

void XGLSharedFBO::CopyOutputToShared(){
	RenderFlipQuad();

	glBindFramebuffer(GL_FRAMEBUFFER, sharedFbo->fbo);
	GL_CHECK("glBindFramebuffer() failed");

	glReadBuffer(GL_COLOR_ATTACHMENT0);
	glReadPixels(0, 0, pHeader->width, pHeader->height, GL_BGR, GL_UNSIGNED_BYTE, mappedBuffer);
	GL_CHECK("glReadPixels() failed\n");

	glBindFramebuffer(GL_FRAMEBUFFER, encoderFbo->fbo);
	GL_CHECK("glBindFramebuffer() failed");

	// enabling GPU profiling below reveals that this takes several milliseconds.
	// Will measure again when async PBO's are in place.
	glReadBuffer(GL_COLOR_ATTACHMENT1);
	glReadPixels(0, 0, encWidth, encHeight, GL_RED, GL_UNSIGNED_BYTE, yBuffer);
	GL_CHECK("glReadPixels() failed\n");

	glReadBuffer(GL_COLOR_ATTACHMENT2);
	glReadPixels(0, 0, encWidth, encHeight, GL_RED, GL_UNSIGNED_BYTE, uBuffer);
	GL_CHECK("glReadPixels() failed\n");

	glReadBuffer(GL_COLOR_ATTACHMENT3);
	glReadPixels(0, 0, encWidth, encHeight, GL_RED, GL_UNSIGNED_BYTE, vBuffer);
	GL_CHECK("glReadPixels() failed\n");

	glBindFramebuffer(GL_FRAMEBUFFER, 0);
	GL_CHECK("glBindFrameBuffer(0) failed");
}

void XGLSharedFBO::Render(int w, int h) {
	vpWidth = w;
	vpHeight = h;

	// set to true for crude GPU profiling.
	if (false){
		GLint64 begin,copy,resolve,scale,read;

		glGetInteger64v(GL_TIMESTAMP, &begin);
		CopyScreenToFBO();
		glGetInteger64v(GL_TIMESTAMP, &copy);
		ResolveMultisampledFBO();
		glGetInteger64v(GL_TIMESTAMP, &resolve);
		ScaleToOutputSize();
		glGetInteger64v(GL_TIMESTAMP, &scale);
		CopyOutputToShared();
		glGetInteger64v(GL_TIMESTAMP, &read);

		printf("%0.3f, %0.3f, %0.3f, %0.3f\n",
		(float)(copy - begin) / 1000000.0,
		(float)(resolve - copy) / 1000000.0,
		(float)(scale - resolve) / 1000000.0,
		(float)(read - scale) / 1000000.0
		);
	}
	else {
		CopyScreenToFBO();
		ResolveMultisampledFBO();
		ScaleToOutputSize();
		CopyOutputToShared();
	}

	if (encoder != NULL)
		encoder->EncodeFrame(mappedBuffer, pHeader->width, pHeader->height, 3);
}