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

	// this is a no-op if there's only one attachment, but if there's more, this is a requirement!
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

#define SAMPLES 8

XGLSharedFBO::XGLSharedFBO() : XSharedMem(DEFAULT_FILE_NAME) {
	xprintf("XGLFramebuffer::XGLFramebuffer()\n");

	glGenFramebuffers(1, &fbo);
	GL_CHECK("glGenFramebuffers() failed");

	glBindFramebuffer(GL_FRAMEBUFFER, fbo);
	GL_CHECK("glBindFramebuffer() failed");

	glGenTextures(1, &texture);
	GL_CHECK("glGenTextures() failed");

	glBindTexture(GL_TEXTURE_2D_MULTISAMPLE, texture);
	GL_CHECK("glBindTexture() failed");

	glTexImage2DMultisample(GL_TEXTURE_2D_MULTISAMPLE, SAMPLES, GL_RGB, RENDER_WIDTH, RENDER_HEIGHT, GL_TRUE);
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

	glRenderbufferStorageMultisample(GL_RENDERBUFFER, SAMPLES, GL_DEPTH_COMPONENT, RENDER_WIDTH, RENDER_HEIGHT);
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

	// now let's generate the output FBO...
	glGenFramebuffers(1, &outFbo);
	GL_CHECK("glGenFramebuffers() failed");

	glBindFramebuffer(GL_FRAMEBUFFER, outFbo);
	GL_CHECK("glBindFramebuffer() failed");

	glGenTextures(1, &outTexture);
	GL_CHECK("glGenTextures() failed");

	glBindTexture(GL_TEXTURE_2D, outTexture);
	GL_CHECK("glBindTexture() failed");

	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, RENDER_WIDTH, RENDER_HEIGHT, 0, GL_RGB, GL_UNSIGNED_BYTE, 0);
	GL_CHECK("glTexImage2D() failed");

	glBindTexture(GL_TEXTURE_2D, 0);
	GL_CHECK("glBindTexture(0) failed");

	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, outTexture, 0);
	GL_CHECK("glFramebufferTexture() failedn");

	if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
		xprintf("glCheckFramebufferStatus() != GL_FRAMEBUFFER_COMPLETE\n");

	glBindFramebuffer(GL_FRAMEBUFFER, 0);
	GL_CHECK("glBindFrameBuffer(0) failed");

	// OpenGL likes 0,0 to be lower left, while the rest of multimedia
	// prefers upper left. Use an XGLTexQuad to do a flip after
	// all has been rendered.
	MakeFlipQuad();

	//encoder = new XAVEncoder();
}

void XGLSharedFBO::MakeFlipQuad() {
	std::string shaderName = pathToAssets + "/shaders/imageflip";
	imgShader = new XGLShader(shaderName);
	imgShader->Compile(shaderName);

	flipQuad = new XGLTexQuad();
	flipQuad->texIds.push_back(intTexture);
	flipQuad->numTextures = 1;
	flipQuad->Load(imgShader, flipQuad->v, flipQuad->idx);
	flipQuad->uniformLocations = imgShader->materialLocations;
	flipQuad->model = glm::translate(glm::mat4(), glm::vec3(0, -0.6667, 0));
}

void XGLSharedFBO::RenderFlipQuad() {
	glBindFramebuffer(GL_FRAMEBUFFER, outFbo);
	glViewport(0, 0, RENDER_WIDTH, RENDER_HEIGHT);
	flipQuad->Render(0.0);
	glViewport(0, 0, vpWidth, vpHeight);
}

void XGLSharedFBO::CopyScreenToFBO(){
	glBindFramebuffer(GL_READ_FRAMEBUFFER, 0);
	GL_CHECK("glBindFrameBuffer(0) failed");

	glBindFramebuffer(GL_DRAW_FRAMEBUFFER, fbo);
	GL_CHECK("glBindFrameBuffer(DRAW) failed");

	// copies the default FBO (the screen) to this FBO
	glBlitFramebuffer(0, 0, vpWidth, vpHeight, 0, 0, vpWidth, vpHeight, GL_COLOR_BUFFER_BIT, GL_LINEAR);
	GL_CHECK("glBlitFramebuffer() failed");
}

void XGLSharedFBO::ResolveMultisampledFBO(){
	glBindFramebuffer(GL_READ_FRAMEBUFFER, fbo);
	GL_CHECK("glBindFrameBuffer(0) failed");

	glBindFramebuffer(GL_DRAW_FRAMEBUFFER, intFbo);
	GL_CHECK("glBindFrameBuffer(DRAW) failed");

	// resolves multi-sampled to single sampled
	glBlitFramebuffer(0, 0, vpWidth, vpHeight, 0, 0, vpWidth, vpHeight, GL_COLOR_BUFFER_BIT, GL_LINEAR);
	GL_CHECK("glBlitFramebuffer() failed");
}

void XGLSharedFBO::ScaleToOutputSize(){
	RenderFlipQuad();
}

void XGLSharedFBO::CopyOutputToShared(){
	glBindFramebuffer(GL_FRAMEBUFFER, outFbo);
	GL_CHECK("glBindFramebuffer() failed");

	glReadPixels(0, 0, pHeader->width, pHeader->height, GL_BGR, GL_UNSIGNED_BYTE, mappedBuffer);
	GL_CHECK("glReadPixels() failed\n");

	glBindFramebuffer(GL_FRAMEBUFFER, 0);
	GL_CHECK("glBindFrameBuffer(0) failed");
}

void XGLSharedFBO::Render(int w, int h) {
	vpWidth = w;
	vpHeight = h;

	CopyScreenToFBO();
	ResolveMultisampledFBO();
	ScaleToOutputSize();
	CopyOutputToShared();

	//encoder->EncodeFrame(mappedBuffer, pHeader->width, pHeader->height, 3);
}