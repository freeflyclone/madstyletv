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

		glTexImage2D(GL_TEXTURE_2D, 0, internalFormat, RENDER_WIDTH, RENDER_HEIGHT, 0, format, GL_UNSIGNED_BYTE, 0);
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

#define SAMPLES 8

XGLSharedFBO::XGLSharedFBO(XGL *context) : XSharedMem(DEFAULT_FILE_NAME), pXGL(context), msFbo(NULL), ssFbo(NULL) {
	// blit only FBO, no depth needed, adding multisampled color attachment
	msFbo = new XGLFramebuffer(RENDER_WIDTH, RENDER_HEIGHT, false, false);

	// create a multi-sampled color buffer for "msFbo"
	glGenTextures(1, &texture);
	GL_CHECK("glGenTextures() failed");
	glBindTexture(GL_TEXTURE_2D_MULTISAMPLE, texture);
	GL_CHECK("glBindTexture() failed");
	glTexImage2DMultisample(GL_TEXTURE_2D_MULTISAMPLE, SAMPLES, GL_RGB, RENDER_WIDTH, RENDER_HEIGHT, GL_TRUE);
	GL_CHECK("glTexImage2D() failed");
	glBindTexture(GL_TEXTURE_2D_MULTISAMPLE, 0);
	GL_CHECK("glBindTexture(0) failed");

	msFbo->AddColorAttachment(texture, GL_TEXTURE_2D_MULTISAMPLE);

	ssFbo = new XGLFramebuffer(RENDER_WIDTH, RENDER_HEIGHT, true, false);
	outFbo = new XGLFramebuffer(RENDER_WIDTH, RENDER_HEIGHT, true, false);

	// add additional color atachments for RGB -> YUV planar for XAVEncoder
	outFbo->AddColorAttachment(0, GL_TEXTURE_2D, GL_RED, GL_R8);
	outFbo->AddColorAttachment(0, GL_TEXTURE_2D, GL_RED, GL_R8);
	outFbo->AddColorAttachment(0, GL_TEXTURE_2D, GL_RED, GL_R8);

	MakeFlipQuad();

	encoder = new XAVEncoder(yBuffer, uBuffer, vBuffer);
}

void XGLSharedFBO::MakeFlipQuad() {
	pXGL->CreateShape("shaders/imageflip", [&]() { flipQuad = new XGLTexQuad(); return flipQuad; });
	flipQuad->AddTexture(ssFbo->textures[0]);
	flipQuad->model = glm::translate(glm::mat4(), glm::vec3(0, -0.6667, 0));
}

void XGLSharedFBO::RenderFlipQuad() {
	glBindFramebuffer(GL_FRAMEBUFFER, outFbo->fbo);
	glViewport(0, 0, RENDER_WIDTH, RENDER_HEIGHT);
	flipQuad->Render(0.0);
	glViewport(0, 0, vpWidth, vpHeight);
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
	RenderFlipQuad();
}

void XGLSharedFBO::CopyOutputToShared(){
	glBindFramebuffer(GL_FRAMEBUFFER, outFbo->fbo);
	GL_CHECK("glBindFramebuffer() failed");

	// enabling GPU profiling below reveals that this takes several milliseconds.
	// Will measure again when async PBO's are in place.
	glReadBuffer(GL_COLOR_ATTACHMENT0);
	glReadPixels(0, 0, pHeader->width, pHeader->height, GL_BGR, GL_UNSIGNED_BYTE, mappedBuffer);
	GL_CHECK("glReadPixels() failed\n");

	glReadBuffer(GL_COLOR_ATTACHMENT1);
	glReadPixels(0, 0, pHeader->width, pHeader->height, GL_RED, GL_UNSIGNED_BYTE, yBuffer);
	GL_CHECK("glReadPixels() failed\n");

	glReadBuffer(GL_COLOR_ATTACHMENT2);
	glReadPixels(0, 0, pHeader->width, pHeader->height, GL_RED, GL_UNSIGNED_BYTE, uBuffer);
	GL_CHECK("glReadPixels() failed\n");

	glReadBuffer(GL_COLOR_ATTACHMENT3);
	glReadPixels(0, 0, pHeader->width, pHeader->height, GL_RED, GL_UNSIGNED_BYTE, vBuffer);
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