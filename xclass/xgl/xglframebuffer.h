/****************************************************************************
**
** Copyright (C) 2015 Evan Mortimore
** All rights reserved.
**
** definitions of OpenGL retained mode API objects:
****************************************************************************/
#ifndef XGLFRAMEBUFFER_H
#define XGLFRAMEBUFFER_H
#include "xglobject.h"
#include "xshmem.h"

typedef std::function<void()> XGLFBORender;

class XGLFramebuffer : public XGLObject {
public:
	XGLFramebuffer(int w, int h, GLuint *texs, int ntex, bool withDepth = true);
	virtual ~XGLFramebuffer();

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
};

class XGLSharedFBO : public XSharedMem {
public:
	XGLSharedFBO();

	virtual void Render(int w, int h);

	XGLFramebuffer *fbo;
	XGLFramebuffer *intFbo;
};

#endif
