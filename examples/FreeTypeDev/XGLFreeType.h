#ifndef XGLFREETYPE_H
#define XGLFREETYPE_H

#include "ft2build.h"
#include FT_OUTLINE_H
#include <vector>
#include "xutils.h"

namespace FT {
	typedef std::vector<FT_Vector> Contour;
	typedef std::vector<Contour> GlyphOutline;

	class GlyphDecomposer : public FT_Outline_Funcs {
	public:
		GlyphDecomposer() {
			move_to = _MoveToFunc;
			line_to = _LineToFunc;
			conic_to = _ConicToFunc;
			cubic_to = _CubicToFunc;
			shift = 0;
			delta = 0;

			// add initial Contour, which is empty, to the outline
			glyphOutline.push_back(*(new Contour()));
		}

		const GlyphOutline& Outline() {
			return glyphOutline;
		}

		bool IsEqual(FT_Vector a, FT_Vector b) { 
			return (a.x == b.x) && (a.y == b.y); 
		}

	private:
		static int _MoveToFunc(const FT_Vector* to, void *pCtx) { return ((GlyphDecomposer*)pCtx)->MoveTo(*to); }
		static int _LineToFunc(const FT_Vector* to, void *pCtx) { return ((GlyphDecomposer*)pCtx)->LineTo(*to); }
		static int _ConicToFunc(const FT_Vector* control, const FT_Vector* to, void *pCtx) { return ((GlyphDecomposer*)pCtx)->ConicTo(*control, *to); }
		static int _CubicToFunc(const FT_Vector* control1, const FT_Vector*	control2, const FT_Vector* to, void *pCtx) { return ((GlyphDecomposer*)pCtx)->CubicTo(*control1, *control2, *to); }

		int MoveTo(const FT_Vector& to);
		int LineTo(const FT_Vector& to);
		int ConicTo(const FT_Vector& control, const FT_Vector& to);
		int CubicTo(const FT_Vector& control1, const FT_Vector& control2, const FT_Vector& to);

		int contourIdx{ 0 };
		GlyphOutline glyphOutline;
		FT_Vector firstPoint;
		FT_Vector currentPoint;
		bool drawCurves = true;
	};



} // namespace FreeType

#endif
