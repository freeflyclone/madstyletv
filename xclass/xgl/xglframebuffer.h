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

#define RENDER_WIDTH	1280
#define RENDER_HEIGHT	720

typedef std::function<void()> XGLFBORender;

class XGLFramebuffer : public XGLObject {
public:
	XGLFramebuffer(int w = RENDER_WIDTH, int h = RENDER_HEIGHT);
	XGLFramebuffer(int w, int h, GLuint *texs, int ntex);
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
};

class XGLSharedFBO : public XGLFramebuffer, public XSharedMem {
public:
	XGLSharedFBO() : XSharedMem(DEFAULT_FILE_NAME) {};
	virtual void Render(XGLFBORender);
};

#endif
