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

class XGLFramebuffer : public XGLObject {
public:
	XGLFramebuffer();
	virtual ~XGLFramebuffer();

	// offscreen MSAA framebuffer
	GLuint fbo;
	GLuint texture;
	GLuint depth;

	// offscreen intermediate framebuffer
	GLuint intFbo;
	GLuint intTexture;
};

class XGLSharedFBO : public XGLFramebuffer, public XSharedMem {
public:
	XGLSharedFBO() : XSharedMem(DEFAULT_FILE_NAME) {};
};

#endif
