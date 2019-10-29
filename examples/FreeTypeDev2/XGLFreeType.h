#ifndef XGLFREETYPE_H
#define XGLFREETYPE_H

#define REAL double
#include "ft2build.h"
#include FT_OUTLINE_H
#include <vector>
#include "xgl.h"
#include "xutils.h"

const XGLTexCoord texCoord{ 0, 1 };
const XGLColor neonYellow = { 0.85, 1.0, 0.00001, 1.0 };
const XGLColor madstyleRed = { 0.85, 0.0, 0.15, 1.0 };

namespace FT {
	class BoundingBox : public std::numeric_limits<float>  {
	public:
		XGLVertex ul, lr;  // upper-left, lower-right corner of the box
	};
	class Contour {
	public:
		Contour() {
			bb.ul = { bb.max(), bb.max(), bb.max() };
			bb.lr = { bb.min(), bb.min(), bb.min() };
		}
		XGLVertex ComputeCentroid();
		void ExpandBoundingBox(XGLVertex vertex);

		XGLVertexList v;
		BoundingBox bb;
		bool isClockwise;
	};
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
		}

		void Reset();

		GlyphOutline& Outline() {
			return glyphOutline;
		}

		bool IsEqual(XGLVertex a, XGLVertex b) {
			return (a.x == b.x) && (a.y == b.y); 
		}

		void DrawCurvesEnable(bool enable) { drawCurves = enable; }

	private:
		static int _MoveToFunc(const FT_Vector* to, void *pCtx) { 
			XGLVertex xto(to->x, to->y, 0);
			return ((GlyphDecomposer*)pCtx)->MoveTo(xto);
		}
		static int _LineToFunc(const FT_Vector* to, void *pCtx) { 
			XGLVertex xto(to->x, to->y, 0);
			return ((GlyphDecomposer*)pCtx)->LineTo(xto);
		}
		static int _ConicToFunc(const FT_Vector* c, const FT_Vector* to, void *pCtx) { 
			XGLVertex xc(c->x, c->y, 0);
			XGLVertex xto(to->x, to->y, 0);
			return ((GlyphDecomposer*)pCtx)->ConicTo(xc, xto);
		}
		static int _CubicToFunc(const FT_Vector* c1, const FT_Vector* c2, const FT_Vector* to, void *pCtx) { 
			XGLVertex xc1(c1->x, c1->y, 0);
			XGLVertex xc2(c2->x, c2->y, 0);
			XGLVertex xto(to->x, to->y, 0);
			return ((GlyphDecomposer*)pCtx)->CubicTo(xc1, xc2, xto); 
		}

		int MoveTo(const XGLVertex&);
		int LineTo(const XGLVertex&);
		int ConicTo(const XGLVertex&, const XGLVertex&);
		int CubicTo(const XGLVertex&, const XGLVertex&, const XGLVertex&);

		const XGLVertex Interpolate(const XGLVertex&, const XGLVertex&, float);
		void EvaluateQuadraticBezier(const XGLVertex&, const XGLVertex&, const XGLVertex&);
		void EvaluateCubicBezier(const XGLVertex&, const XGLVertex&, const XGLVertex&, const XGLVertex&);

		GlyphOutline glyphOutline;
		int contourIdx{ 0 };
		Contour* currentContour;
		XGLVertex firstPoint;
		XGLVertex currentPoint;
		REAL interpolationFactor = 0.1;

		bool drawCurves{ true };
		bool useColors{ true };
	};
} // namespace FT

// this must be after namespace FT, as Triangulator class depends on it.
#include "Triangulator.h"

#endif
