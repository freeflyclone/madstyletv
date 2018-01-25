/****************************************************************************
**
** Copyright (C) 2015 Evan Mortimore
** All rights reserved.
**
** definitions of OpenGL retained mode API objects:
****************************************************************************/
#ifndef XGLPIXELBUFFER_H
#define XGLPIXELBUFFER_H
#include "xshmem.h"

typedef std::function<void()> XGLPBORender;

class XGLPixelbuffer : public XObject {
public:
	XGLPixelbuffer(int w = XGLFramebuffer::renderWidth, int h = XGLFramebuffer::renderHeight);
	virtual ~XGLPixelbuffer();

	virtual void Render(unsigned char *);

	int width, height;
};

class XGLSharedPBO : public XGLPixelbuffer, public XSharedMem {
public:
	XGLSharedPBO() : XSharedMem(shmemDefaultFile) {};

	virtual void Render();
};

#endif
