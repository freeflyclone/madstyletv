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

		void DrawCurves(bool enable) { drawCurves = enable; }

	private:
		static int _MoveToFunc(const FT_Vector* to, void *pCtx) { return ((GlyphDecomposer*)pCtx)->MoveTo(*to); }
		static int _LineToFunc(const FT_Vector* to, void *pCtx) { return ((GlyphDecomposer*)pCtx)->LineTo(*to); }
		static int _ConicToFunc(const FT_Vector* c, const FT_Vector* to, void *pCtx) { return ((GlyphDecomposer*)pCtx)->ConicTo(*c, *to); }
		static int _CubicToFunc(const FT_Vector* c1, const FT_Vector* c2, const FT_Vector* to, void *pCtx) { return ((GlyphDecomposer*)pCtx)->CubicTo(*c1, *c2, *to); }

		int MoveTo(const FT_Vector&);
		int LineTo(const FT_Vector&);
		int ConicTo(const FT_Vector&, const FT_Vector&);
		int CubicTo(const FT_Vector&, const FT_Vector&, const FT_Vector&);

		const FT_Vector& Interpolate(const FT_Vector&, const FT_Vector&, float);
		void EvaluateQuadraticBezier(const FT_Vector&, const FT_Vector&, const FT_Vector&);
		void EvaluateCubicBezier(const FT_Vector&, const FT_Vector&, const FT_Vector&, const FT_Vector&);

		GlyphOutline glyphOutline;
		int contourIdx{ 0 };
		FT_Vector firstPoint;
		FT_Vector currentPoint;
		bool drawCurves = true;
		float interpolationFactor = 0.1f;
	};
} // namespace FT

#endif
