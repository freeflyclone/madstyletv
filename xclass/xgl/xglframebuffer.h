/****************************************************************************
**
** Copyright (C) 2016 Evan Mortimore
** All rights reserved.
**
** definitions of OpenGL retained mode API objects:
****************************************************************************/
#ifndef XGLFRAMEBUFFER_H
#define XGLFRAMEBUFFER_H
#include "xglobject.h"
#include "xglprimitives.h"
#include "xshmem.h"
#include "xavenc.h"

// the dimensions of the FBO buffer images.  This is GPU video memory.
// You've been advised. Tested to 3840 by 2160
#define RENDER_WIDTH	1920
#define RENDER_HEIGHT	1080

typedef std::function<void()> XGLFBORender;

class XGLFramebuffer : public XGLObject {
public:
	XGLFramebuffer(int w, int h, bool withColor = true, bool withDepth = true, GLuint texId = 0);
	virtual ~XGLFramebuffer();

	void AddColorAttachment(GLuint texId=0, GLenum target=GL_TEXTURE_2D, GLint format=GL_RGB, GLenum internalFormat = GL_RGB);
	void AddDepthBuffer();

	virtual void Render(XGLFBORender);

	// offscreen framebuffer
	GLuint fbo;
	GLuint textures[8];
	GLuint depth;
	GLuint attachments[8];

	// dimensions (used for glViewport())
	int width, height;
	int numTextures;

	// an FBO being used as a destination for glBlitFramebuffer()
	// (for example, from system FBO to shared memory) doesn't
	// require depth buffer, only color.
	bool hasDepth;

	// it's also possible that no color attachments are desired. (I think)
	bool hasColor;
};

class XGLSharedFBO : public XSharedMem {
public:
	XGLSharedFBO(XGL *context);

	// this gets called at then end of the XGL::Display()
	virtual void Render(int width, int height);

	void CopyScreenToFBO();
	void ResolveMultisampledFBO();
	void ScaleToOutputSize();
	void CopyOutputToShared();

	void MakeFlipQuad();
	void RenderFlipQuad();

	XGL *pXGL;
	XGLFramebuffer *msFbo;
	XGLFramebuffer *ssFbo;
	XGLFramebuffer *scaleFbo;
	XGLFramebuffer *sharedFbo;
	XGLFramebuffer *encoderFbo;
	GLuint texture;

	XGLTexQuad *flipQuad;
	XGLShader *imgShader;

	// dimensions to restore to after rendering the flipQuad
	int vpWidth, vpHeight;

	XAVEncoder *encoder;
	int encWidth, encHeight;

	// TODO: manage these better for systems with reduced memory available.
	// (who has that these days and why are they trying to run this code?)
	unsigned char yBuffer[RENDER_WIDTH*RENDER_HEIGHT];
	unsigned char uBuffer[RENDER_WIDTH*RENDER_HEIGHT];
	unsigned char vBuffer[RENDER_WIDTH*RENDER_HEIGHT];
};

#endif
