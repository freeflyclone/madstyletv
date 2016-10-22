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
	virtual ~XGLFramebuffer();

	virtual void Render(XGLFBORender);

	// offscreen MSAA framebuffer
	GLuint fbo;
	GLuint texture;
	GLuint depth;

	// dimensions (used for glViewport())
	int width, height;
};

class XGLSharedFBO : public XGLFramebuffer, public XSharedMem {
public:
	XGLSharedFBO() : XSharedMem(DEFAULT_FILE_NAME) {};
	virtual void Render(XGLFBORender);
};

#endif
