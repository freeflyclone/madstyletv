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
#define FILEMAPPING_SIZE ((1920*1080*4)+4096)

typedef struct {
	unsigned int width;
	unsigned int height;
	unsigned int bytesPerPixel;
	unsigned char reserved[4084];
} MAPPED_HEADER;

class XGLFramebuffer : public XGLObject {
public:
	XGLFramebuffer();
	virtual ~XGLFramebuffer();

	GLuint fbo;
	GLuint texture;
	GLuint depthrenderbuffer;
	HANDLE hFile, hMapping;
	unsigned char *mappedHeader;
	unsigned char *mappedBuffer;
	MAPPED_HEADER *pHeader;
};


#endif
