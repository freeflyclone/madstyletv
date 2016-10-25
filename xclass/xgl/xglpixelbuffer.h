/****************************************************************************
**
** Copyright (C) 2015 Evan Mortimore
** All rights reserved.
**
** definitions of OpenGL retained mode API objects:
****************************************************************************/
#ifndef XGLPIXELBUFFER_H
#define XGLPIXELBUFFER_H
#include "xglobject.h"
#include "xshmem.h"

#define RENDER_WIDTH	1280
#define RENDER_HEIGHT	720

typedef std::function<void()> XGLPBORender;

class XGLPixelbuffer : public XGLObject {
public:
	XGLPixelbuffer(int w = RENDER_WIDTH, int h = RENDER_HEIGHT);
	virtual ~XGLPixelbuffer();

	virtual void Render(unsigned char *);

	int width, height;
};

class XGLSharedPBO : public XGLPixelbuffer, public XSharedMem {
public:
	XGLSharedPBO() : XSharedMem(DEFAULT_FILE_NAME) {};

	virtual void Render();
};

#endif
