#ifndef XGLPIXELFORMAT_H
#define XGLPIXELFORMAT_H
#include "XGL.h"

// a class for OpenGL TexImage texture map images
class XGLPixelFormatDescriptor : public AVPixFmtDescriptor {
public:
	typedef std::map<GLint, int> InternalFormatSizes;
	XGLPixelFormatDescriptor(AVPixelFormat pf) {
		ifs[GL_R8] = 1;
		ifs[GL_RG] = 2;
		ifs[GL_RGB] = 3;
		ifs[GL_RGBA] = 4;

		const AVPixFmtDescriptor *pixDesc = av_pix_fmt_desc_get((AVPixelFormat)pf);

		// if pixDesc describes a YUV format
		if (!(pixDesc->flags&AV_PIX_FMT_FLAG_RGB)) {
			int idx = 0;
			for (auto c : pixDesc->comp)
				depths[idx++] = c.depth / 8;

			internalFormat = GL_RED;
			format = GL_RED;
			type = GL_UNSIGNED_BYTE;

			shiftRightH = pixDesc->log2_chroma_h;
			shiftRightW = pixDesc->log2_chroma_w;
			nPlanes = pixDesc->nb_components;
		}
		else
			throwXGLException("XGLPixelFormatDescriptor() doesn't to RGB yet");
	}

	int PixelSize() { return ifs[internalFormat]; }

	GLint internalFormat;	// #color components, GL_RED, GL_RG, GL_RGB, GL_RGBA are preferred
	GLenum format;			// format of pixel data
	GLenum type;			// data type of pixel data
	GLint nPlanes;
	GLint depths[4];
	GLint shiftRightH, shiftRightW;
	InternalFormatSizes ifs;
};

typedef std::map<AVPixelFormat, XGLPixelFormatDescriptor> XGLPixelFormatDescriptors;
#endif