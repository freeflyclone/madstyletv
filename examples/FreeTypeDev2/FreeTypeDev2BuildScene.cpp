/**************************************************************
** FreeTypeDev2BuildScene.cpp
**
** Second Freetype rendering experiment.  First one involved
** using "Triangle": a super old school C library for outline
** triangulation.  Great code, but major overkill for fonts.
**
** Previous attempt was expensive CPU triangulation after CPU
** bezier interpolation, with excessive interpolation to yield
** acceptable visual quality. Not good for interactive updates.
**
** This attempt is focusing on a hybrid approach:
** simple (hopefully) CPU triangulation of TTF font outline points
** with GPU shader for bezier curve interpolation.
**************************************************************/
#include "ExampleXGL.h"
#include <string>
#include "XGLFreeType.h"

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

		// scale so outline coordinate precision is adequately preserved
		FT_Set_Char_Size(face, ftSize, ftSize, ftResolution, ftResolution);

		for (charcode = FT_Get_First_Char(face, &gindex); gindex; charcode = FT_Get_Next_Char(face, charcode, &gindex))
			charMap.emplace(charcode, gindex);

		// add signal to initial Load() to initialize the materials properties.
		v.push_back({});
	}

	void PushVertex(XGLVertexAttributes vrtx) {
		vrtx.v.x /= scaleFactor;
		vrtx.v.y /= scaleFactor;
		
		vrtx.v.x += advance.x;
		vrtx.v.y += advance.y;

		vrtx.c = XGLColors::yellow;

		v.push_back(vrtx);
	}

	void AdvanceGlyphPosition() {
		advance.x += face->glyph->advance.x / scaleFactor;
		advance.y += face->glyph->advance.y / scaleFactor;
	}

	void RenderText(std::string textToRender) {
		v.clear();
		contourOffsets.clear();

		advance = { 0.0f, 0.0f, 0.0f };

		for (char c : textToRender) {
			FT_Load_Glyph(face, charMap[c], FT_LOAD_FORCE_AUTOHINT | FT_LOAD_TARGET_NORMAL);
			Reset();
			FT_Outline_Decompose(&face->glyph->outline, (FT_Outline_Funcs*)this, (FT_Outline_Funcs*)this);

			for (auto c : Outline()) {
				contourOffsets.push_back((int)v.size());
				for (auto vrtx : c.v)
					PushVertex(vrtx);
			}
			// mark end offset, so display loop can calculate size
			contourOffsets.push_back((int)v.size());
			AdvanceGlyphPosition();
		}

		// update the VBO with new geometry
		Load(shader, v, idx);
	}

	void Draw() {
		if (contourOffsets.size()) {
			for (int idx = 0; idx < contourOffsets.size() - 1; idx++) {
				GLuint start = contourOffsets[idx];
				GLuint length = contourOffsets[idx + 1] - start;

				glDrawArrays(GL_LINE_LOOP, start, length);
				GL_CHECK("glDrawArrays() failed");
			}
		}
	}

private:
	// These 2 numbers help Freetype math have adequate precision...
	const FT_F26Dot6 ftSize{ 1024 };
	const FT_UInt ftResolution{ 1024 };

	// ...while this is used to scale back to XGL preferred size
	REAL scaleFactor{ 3276.8f };

	FT_Library ft;
	FT_Face face;
	FT_GlyphSlot g;
	CharMap charMap;
	XGLVertex advance;
	std::vector<int>contourOffsets;
};

void ExampleXGL::BuildScene() {
	XGLFreeType *shape;

	AddShape("shaders/000-simple", [&](){ shape = new XGLFreeType(); return shape; });

	shape->RenderText("MadStyle TV");
}
