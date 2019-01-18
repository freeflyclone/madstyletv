/**************************************************************
** FreeTypeDevBuildScene.cpp
**
** Demonstrate drawing FreeType outlines using 
** FT_Outline_Decompose. Found the EvaluateXXXBezier() methods
** on StackOverFlow.
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

	class FreeTypeDecomposer : public FT_Outline_Funcs {
	public:
		static int MoveToFunc(const FT_Vector* to, void *pCtx) {
			XGLFreeType* pxft = (XGLFreeType*)pCtx;
			pxft->v.push_back({ { to->x / pxft->scaleFactor, to->y / pxft->scaleFactor, 0 }, {}, {}, { 1, 1, 0, 1 } });
			pxft->fdc.currentPoint = *to;
			return 0;
		}
		static int LineToFunc(const FT_Vector* to, void *pCtx) {
			XGLFreeType* pxft = (XGLFreeType*)pCtx;
			pxft->v.push_back({ { to->x / pxft->scaleFactor, to->y / pxft->scaleFactor, 0 }, {}, {}, { 1, 1, 0, 1 } });
			pxft->fdc.currentPoint = *to;
			return 0;
		}
		static int ConicToFunc(const FT_Vector*	control, const FT_Vector* to, void *pCtx){
			XGLFreeType* pxft = (XGLFreeType*)pCtx;
			pxft->EvaluateQuadraticBezier(pxft->fdc.currentPoint, *control, *to);
			pxft->fdc.currentPoint = *to;
			return 0;
		}
		static int CubicToFunc(const FT_Vector*	control1, const FT_Vector*	control2, const FT_Vector* to, void *pCtx){
			XGLFreeType* pxft = (XGLFreeType*)pCtx;
			pxft->EvaluateCubicBezier(pxft->fdc.currentPoint, *control1, *control2, *to);
			pxft->fdc.currentPoint = *to;
			return 0;
		}

		FreeTypeDecomposer() {
			move_to = MoveToFunc;
			line_to = LineToFunc;
			conic_to = ConicToFunc;
			cubic_to = CubicToFunc;
			shift = 0;
			delta = 0;
		}

		FT_Vector currentPoint;
	};

	FreeTypeDecomposer fdc;

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

		gindex = charMap['&'];

		FT_Load_Glyph(face, gindex, FT_LOAD_FORCE_AUTOHINT | FT_LOAD_TARGET_LIGHT);

		FT_Outline fto = g->outline;
		FT_Outline_Decompose(&fto, &fdc, this);
	};

	~XGLFreeType() {
		FT_Done_Face(face);
		FT_Done_FreeType(ft);
	};

	void Draw() {
		if (v.size()) {
			glDrawArrays(GL_LINE_LOOP, 0, (GLsizei)v.size());
			GL_CHECK("glDrawArrays() failed");
		}
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
