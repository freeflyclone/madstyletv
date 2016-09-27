/****************************************************************************
**
** Copyright (C) 2015 Evan Mortimore
** All rights reserved.
**
** definitions of OpenGL retained mode API objects:
****************************************************************************/
#ifndef XGLFRAMEBUFFER_H
#define XGLFRAMEBUFFER_H

#include "XGLObject.h"

#define RENDER_WIDTH	1280
#define RENDER_HEIGHT	720

class XGLFramebuffer : public XGLObject {
public:
	XGLFramebuffer();
	virtual ~XGLFramebuffer();

	GLuint fbo;
	GLuint texture;
	GLuint depthrenderbuffer;
};


#endif
