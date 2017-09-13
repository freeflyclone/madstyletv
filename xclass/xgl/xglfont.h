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
#define FONT_NAME "C:/windows/fonts/Arial.ttf"
#elif __APPLE__
#include "TargetConditionals.h"
#if TARGET_OS_MAC
#define FONT_NAME "/Library/Fonts/Arial.ttf"
#endif
#elif __linux__
#define FONT_NAME "/usr/share/fonts/truetype/freefont/FreeSans.ttf"
#endif

class XGLFont {
public:
	// define everything we may need in order to render glyphs
	// directly from pages of bitmap textures
	struct  XGLGlyph {
		FT_UInt index;		// In case we need to call FT_Load_Glyph
		int	page;			// the atlas page for this glyph's bitmap
		int row;			// the row on that page
		int xOff, yOff;		// the X,Y coords of the upper left corner
		int width, height;	// it's dimensions
	};

    XGLFont();
    ~XGLFont();

	XGLGlyph *GetGlyph(int c) { return &glyphMap[c]; }
	void RenderText(std::wstring, unsigned char *, int, int, int *, int *);
	void RenderText(std::string, unsigned char *, int, int, int *, int *);
	void SetPixelSize(int);

	int MeasureStringWidth(std::wstring) const;
	int MeasureStringWidth(std::string) const;
	int MeasureFontHeight() const;
	int MeasureBaselineHeight() const;

	typedef std::map<FT_ULong, FT_UInt> CharMap;
	typedef std::map<FT_ULong, XGLGlyph> GlyphMap;

	FT_Library ft;
    FT_Face face;
    FT_GlyphSlot g;

    CharMap charMap;
	GlyphMap glyphMap;

    // there may be many pages (2D bitmaps) of
    // texture atlasses, particularly for
    // Chinese fonts.
    GLubyte **bitmapPages;

    GLuint atlasWidth;
    GLuint atlasHeight;
    GLuint atlasPageCount;
	int maxAscend, maxDescend;
	int pixelSize;
};

#endif
