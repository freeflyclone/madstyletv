/**************************************************************
** FreeTypeDev2BuildScene.cpp
**
** Just to demonstrate instantiation of a "ground"
** plane and a single triangle, with default camera manipulation
** via keyboard and mouse.
**************************************************************/
#include "ExampleXGL.h"
#include <string>
#include "XGLFreeType.h"

#define REAL double
#include FT_OUTLINE_H

#ifdef FONT_NAME
#undef FONT_NAME
#endif

#define FONT_NAME "C:/windows/fonts/times.ttf"

class XGLFreeType : public FT::GlyphDecomposer,  public XGLShape {
public:
	typedef std::map<FT_ULong, FT_UInt> CharMap;
	typedef std::vector<GLsizei> ContourEndPoints;

	XGLFreeType() {
		FT_UInt gindex = 0;
		FT_ULong charcode = 0;

		if (FT_Init_FreeType(&ft))
			throwXGLException("Init of FreeType failed");

		if (FT_New_Face(ft, FONT_NAME, 0, &face))
			throwXGLException("FT_New_Face() failed " FONT_NAME);

		if (FT_Select_Charmap(face, ft_encoding_unicode))
			throwXGLException("FT_Select_Charmap(UNICODE) failed.");

		// scaled so outline coordinates have viable values
		FT_Set_Char_Size(face, ftSize, ftSize, ftResolution, ftResolution);

		for (charcode = FT_Get_First_Char(face, &gindex); gindex; charcode = FT_Get_Next_Char(face, charcode, &gindex))
			charMap.emplace(charcode, gindex);
	}

	void RenderText(std::string textToRender) {
		xprintf("%s(): %s\n", __FUNCTION__, textToRender.c_str());

		for (auto c : textToRender) {
			FT_Load_Glyph(face, charMap[c], FT_LOAD_FORCE_AUTOHINT | FT_LOAD_TARGET_NORMAL);
			Reset();
			FT_Outline_Decompose(&face->glyph->outline, (FT_Outline_Funcs*)this, (FT_Outline_Funcs*)this);

			for (auto c : Outline()) {
				xprintf("contour bb: {%0.4f, %0.4f}, {%0.4f, %0.4f}\n", c.bb.ul.x, c.bb.ul.y, c.bb.lr.x, c.bb.lr.y);
			}
		}
	}

	void Draw() {
	}

private:
	const FT_F26Dot6 ftSize{ 1024 };
	const FT_UInt ftResolution{ 1024 };
	FT_Library ft;
	FT_Face face;
	FT_GlyphSlot g;
	CharMap charMap;
};

void ExampleXGL::BuildScene() {
	XGLFreeType *shape;

	AddShape("shaders/000-simple", [&](){ shape = new XGLFreeType(); return shape; });
	shape->RenderText("A");
}
