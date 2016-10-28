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
#include "xshmem.h"
#include "xavenc.h"

#define RENDER_WIDTH	1920
#define RENDER_HEIGHT	1080

typedef std::function<void()> XGLFBORender;

class XGLFramebuffer : public XGLObject {
public:
	XGLFramebuffer(int w, int h, bool withColor = true, bool withDepth = true, GLuint texId = 0);
	virtual ~XGLFramebuffer();

	void AddColorAttachment(GLuint texId=0, GLenum target=GL_TEXTURE_2D);
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
	XGLSharedFBO();

	// this gets called at then end of the XGL::Display()
	virtual void Render(int width, int height);

	void CopyScreenToFBO();
	void ResolveMultisampledFBO();
	void ScaleToOutputSize();
	void CopyOutputToShared();

	void MakeFlipQuad();
	void RenderFlipQuad();

	XGLFramebuffer *msFbo;
	XGLFramebuffer *ssFbo;
	XGLFramebuffer *outFbo;
	GLuint texture;

	XGLTexQuad *flipQuad;
	XGLShader *imgShader;

	// dimensions to restore to after rendering the flipQuad
	int vpWidth, vpHeight;

	XAVEncoder *encoder;
};

#endif
