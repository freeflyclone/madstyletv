/**************************************************************
** FreeTypeDevBuildScene.cpp
**
** Just to demonstrate instantiation of a "ground"
** plane and a single triangle, with default camera manipulation
** via keyboard and mouse.
**************************************************************/
#include "ExampleXGL.h"

#include FT_OUTLINE_H

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
		FT_Outline_Funcs ftof;

		ftof.move_to = [](const FT_Vector* to, void* user) -> int { xprintf("MoveToFunc\n");  return 0; };
		ftof.line_to = [](const FT_Vector* to, void* user) -> int { xprintf("LineToFunc\n");  return 0; };
		ftof.conic_to = [](const FT_Vector* control, const FT_Vector*to, void* user) -> int { xprintf("ConicToFunc\n");  return 0; };
		ftof.cubic_to = [](const FT_Vector* control1, const FT_Vector* control2, const FT_Vector*to, void* user) -> int { xprintf("CubicToFunc\n");  return 0; };

		FT_Outline_Decompose(&fto, &ftof, this);

		if (false) {
			xprintf("    format: %c%c%c%c\n", (g->format >> 24) & 0xFF, (g->format >> 16) & 0xFF, (g->format >> 8) & 0xFF, (g->format) & 0xFF);
			xprintf("n_contours: %d\n", fto.n_contours);
			xprintf("  n_points: %d\n", fto.n_points);

			FT_Vector* pftv;
			int i;
			int count = fto.n_points;
			char *pTags = fto.tags;

			for (i = 0, pftv = fto.points; i < count; i++, pftv++, pTags++) {
				xprintf("Point %d: (%d,%d) - %02X\n", i, pftv->x, pftv->y, fto.tags[i] & 0xFF);
				if (i + 1 < count) {
					if (*(pTags + 1) & 1)
						v.push_back({ { pftv->x / scaleFactor, pftv->y / scaleFactor, 0 }, {}, {}, { 1, 1, 0, 1 } });
					else if (((*(pTags + 1) & 1) == 0) && ((*(pTags + 2) & 1) == 0)) {
						EvaluateCubicBezier(*pftv, *(pftv + 1), *(pftv + 2), *(pftv + 3));
						i += 2;
						pftv += 2;
						pTags += 2;
					}
					else if (((*(pTags + 1) & 1) == 0) && ((*(pTags + 2) & 1) == 1)) {
						EvaluateQuadraticBezier(*pftv, *(pftv + 1), *(pftv + 2));
						i += 1;
						pftv += 1;
						pTags += 1;
					}
				}
				else
					v.push_back({ { pftv->x / scaleFactor, pftv->y / scaleFactor, 0 }, {}, {}, { 1, 1, 0, 1 } });
			}
		}
	};

	~XGLFreeType() {
		FT_Done_Face(face);
		FT_Done_FreeType(ft);
	};

	void Draw() {
		if (v.size() == 0)
			return;
		glDrawArrays(GL_LINE_LOOP, 0, (GLsizei)v.size());
		GL_CHECK("glDrawArrays() failed");
	}

	float GetInterpolatedPoint(float n1, float n2, float percent) {
		float diff = n2 - n1;
		return n1 + (diff * percent);
	}

	void EvaluateQuadraticBezier(FT_Vector p0, FT_Vector p1, FT_Vector p2) {
		xprintf("EvaluateQuadratic\n");
		float xa, xb, ya, yb;
		float x, y;
		float interpolant;

		for (interpolant = 0.0f; interpolant < 1.0f; interpolant += 0.1f) {
			xa = GetInterpolatedPoint((float)p0.x, (float)p1.x, interpolant);
			ya = GetInterpolatedPoint((float)p0.y, (float)p1.y, interpolant);

			xb = GetInterpolatedPoint((float)p1.x, (float)p2.x, interpolant);
			yb = GetInterpolatedPoint((float)p1.y, (float)p2.y, interpolant);

			x = GetInterpolatedPoint(xa, xb, interpolant);
			y = GetInterpolatedPoint(ya, yb, interpolant);

			v.push_back({ { x/scaleFactor, y/scaleFactor, 0 }, {}, {}, { 1, 1, 0, 1 } });
		}
	}

	void EvaluateCubicBezier(FT_Vector p0, FT_Vector p1, FT_Vector p2, FT_Vector p3) {
		xprintf("EvaluateCubic\n");
		float xa, xb, xc, ya, yb, yc;
		float xm, xn, ym, yn;
		float x, y;
		float interpolant;

		for (interpolant = 0.0f; interpolant < 1.0f; interpolant += 0.1f) {
			xa = GetInterpolatedPoint((float)p0.x, (float)p1.x, interpolant);
			ya = GetInterpolatedPoint((float)p0.y, (float)p1.y, interpolant);
			xb = GetInterpolatedPoint((float)p1.x, (float)p2.x, interpolant);
			yb = GetInterpolatedPoint((float)p1.y, (float)p2.y, interpolant);
			xc = GetInterpolatedPoint((float)p2.x, (float)p3.x, interpolant);
			yc = GetInterpolatedPoint((float)p2.y, (float)p3.y, interpolant);

			xm = GetInterpolatedPoint(xa, xb, interpolant);
			ym = GetInterpolatedPoint(ya, yb, interpolant);
			xn = GetInterpolatedPoint(xb, xc, interpolant);
			yn = GetInterpolatedPoint(yb, yc, interpolant);

			x = GetInterpolatedPoint(xm, xn, interpolant);
			y = GetInterpolatedPoint(ym, yn, interpolant);

			v.push_back({ { x / scaleFactor, y / scaleFactor, 0 }, {}, {}, { 1, 1, 0, 1 } });
		}
	}

	FT_Library ft;
	FT_Face face;
	FT_GlyphSlot g;
	CharMap charMap;
	float scaleFactor = 200.0f;
};

void ExampleXGL::BuildScene() {
	XGLFreeType *shape;

	AddShape("shaders/000-simple", [&](){ shape = new XGLFreeType(); return shape; });
}
