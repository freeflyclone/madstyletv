/**************************************************************
** FreeTypeDevBuildScene.cpp
**
** Just to demonstrate instantiation of a "ground"
** plane and a single triangle, with default camera manipulation
** via keyboard and mouse.
**************************************************************/
#include "ExampleXGL.h"

#ifdef FONT_NAME
#undef FONT_NAME
#endif

#define FONT_NAME "C:/windows/fonts/times.ttf"

class XGLFreeType : public XGLShape {
public:
	typedef std::map<FT_ULong, FT_UInt> CharMap;

	XGLFreeType() {
		FT_UInt gindex = 0;
		FT_ULong charcode = 0;

		if (FT_Init_FreeType(&ft))
			throwXGLException("Init of FreeType failed");

		if (FT_New_Face(ft, FONT_NAME, 0, &face))
			throwXGLException("FT_New_Face() failed " FONT_NAME);

		if (FT_Select_Charmap(face, ft_encoding_unicode))
			throwXGLException("FT_Select_Charmap(UNICODE) failed.");

		// scale the rendering to 1 pixel per 'point' resolution
		FT_Set_Char_Size(face, 256, 0, 512, 0);

		FT_GlyphSlot g = face->glyph;

		// build an XGLCharMap of the entire set of glyphs for this font.
		// (this could be huge for Chinese fonts)
		for (charcode = FT_Get_First_Char(face, &gindex); gindex; charcode = FT_Get_Next_Char(face, charcode, &gindex))
			charMap.emplace(charcode, gindex);

		const int numGlyphs = (const int)(charMap.size());
		xprintf("XGLFreeType::XGLFreeType() - There are %d glyphs.\n", numGlyphs);

		gindex = charMap['M'];

		FT_Load_Glyph(face, gindex, FT_LOAD_FORCE_AUTOHINT | FT_LOAD_TARGET_LIGHT);

		FT_Outline fto = g->outline;

		xprintf("    format: %c%c%c%c\n", (g->format >> 24) & 0xFF, (g->format >> 16) & 0xFF, (g->format >> 8) & 0xFF, (g->format) & 0xFF);
		xprintf("n_contours: %d\n", fto.n_contours);
		xprintf("  n_points: %d\n", fto.n_points);

		FT_Vector* pftv;
		int i;
		for (i = 0, pftv = fto.points; i < fto.n_points; i++, pftv++) {
			xprintf("Point %d: (%d,%d) - %02X\n", i, pftv->x, pftv->y, fto.tags[i] & 0xFF);
			v.push_back({ { pftv->x / 500.0f, pftv->y / 500.0f, 0 }, {}, {}, { 1, 1, 0, 1 } });
		}
	};

	~XGLFreeType() {
		FT_Done_Face(face);
		FT_Done_FreeType(ft);
	};

	void Draw() {
		glDrawArrays(GL_LINE_LOOP, 0, (GLsizei)v.size());
		GL_CHECK("glDrawArrays() failed");
	}

	FT_Library ft;
	FT_Face face;
	FT_GlyphSlot g;
	CharMap charMap;
};

void ExampleXGL::BuildScene() {
	XGLFreeType *shape;

	AddShape("shaders/000-simple", [&](){ shape = new XGLFreeType(); return shape; });
}
