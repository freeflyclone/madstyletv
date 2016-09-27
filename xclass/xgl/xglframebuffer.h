/****************************************************************************
**
** Copyright (C) 2015 Evan Mortimore
** All rights reserved.
**
** definitions of OpenGL retained mode API objects:
****************************************************************************/
#ifndef XGLFRAMEBUFFER_H
#define XGLFRAMEBUFFER_H

#include <Windows.h>

#include "XGLObject.h"

#define RENDER_WIDTH	1280
#define RENDER_HEIGHT	720
#define FILEMAPPING_SIZE (1920*1080*4)

class XGLFramebuffer : public XGLObject {
public:
	XGLFramebuffer();
	virtual ~XGLFramebuffer();

	GLuint fbo;
	GLuint texture;
	GLuint depthrenderbuffer;
	HANDLE hFile, hMapping;
	unsigned char *fileMappedBuffer;
};


#endif
