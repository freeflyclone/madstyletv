/****************************************************************************
**
** Copyright (C) 2016 Evan Mortimore
** All rights reserved.
**
** definitions of OpenGL <-> FreeType objects:
**
****************************************************************************/
#ifndef XGLFONT_H
#define XGLFONT_H

#include "xgl.h"

#ifdef WIN32
//#define FONT_NAME "/windows/fonts/mingliub.ttc"
#define FONT_NAME "/windows/fonts/YuGothic.ttf"
#elif __APPLE__
#include "TargetConditionals.h"
#if TARGET_OS_MAC
#define FONT_NAME "/Library/Fonts/Arial.ttf"
#endif
#elif __linux__
#endif

class XGLFont {
    typedef std::map<FT_ULong, FT_UInt> CharMap;

public:

    XGLFont();
    ~XGLFont();

    FT_Library ft;
    FT_Face face;
    FT_GlyphSlot g;

    CharMap charMap;

    // there may be many pages (2D bitmaps) of
    // texture atlasses, particularly for
    // Chinese fonts.
    GLubyte **bitmapPages;

    GLuint tex;
    GLuint vbo;
    GLuint atlasWidth;
    GLuint atlasHeight;
    GLuint atlasPageCount;
};

#endif