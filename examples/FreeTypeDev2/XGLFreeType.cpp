#include "XGLFreeType.h"

using namespace FT;

XGLVertex Contour::ComputeCentroid()
{
	XGLVertex centroid = {};

	// upper layer may be outlining a whitespace char which will have no PSLG outline.
	// As one might imagine. ;)
	if (v.size() == 0)
		return centroid;

	REAL signedArea = 0.0;
	REAL x0 = 0.0; // Current vertex X
	REAL y0 = 0.0; // Current vertex Y
	REAL x1 = 0.0; // Next vertex X
	REAL y1 = 0.0; // Next vertex Y
	REAL a = 0.0;  // Partial signed area

	// For all vertices except last
	int i = 0;
	for (i = 0; i<v.size() - 1; ++i)
	{
		x0 = v[i].v.x;
		y0 = v[i].v.y;
		x1 = v[i+1].v.x;
		y1 = v[i + 1].v.y;
		a = x0*y1 - x1*y0;
		signedArea += a;
		centroid.x += (x0 + x1)*a;
		centroid.y += (y0 + y1)*a;
	}

	// Do last vertex separately to avoid performing an expensive
	// modulus operation in each iteration.
	x0 = v[i].v.x;
	y0 = v[i].v.y;
	x1 = v[0].v.x;
	y1 = v[0].v.y;
	a = x0*y1 - x1*y0;
	signedArea += a;
	centroid.x += (x0 + x1)*a;
	centroid.y += (y0 + y1)*a;

	signedArea *= 0.5;
	centroid.x /= (6.0*signedArea);
	centroid.y /= (6.0*signedArea);

	if (signedArea < 0)
		isClockwise = true;
	else
		isClockwise = false;

	return centroid;
}

void Contour::ExpandBoundingBox(XGLVertex vertex) {
	if (vertex.x < bb.ul.x)
		bb.ul.x = vertex.x;
	if (vertex.y < bb.ul.y)
		bb.ul.y = vertex.y;
	if (vertex.x > bb.lr.x)
		bb.lr.x = vertex.x;
	if (vertex.y > bb.lr.y)
		bb.lr.y = vertex.y;
}

void GlyphDecomposer::Reset() {
	for (FT::Contour c : glyphOutline)
		c.v.clear();

	glyphOutline.clear();
	glyphOutline.push_back(*(new Contour()));
	currentContour = &glyphOutline.back();
	contourIdx = 0;
}

int GlyphDecomposer::MoveTo(const XGLVertex& to) {
	//xprintf(" MoveTo: %0.4f %0.4f\n", to.x, to.y);
	firstPoint = to;

	// if the current Contour has XGLVertex data, this MoveTo is a new Contour
	if (glyphOutline[contourIdx].v.size()) {
		contourIdx++;
		glyphOutline.push_back(*(new Contour()));
		currentContour = &glyphOutline.back();
	}

	glyphOutline[contourIdx].v.push_back({ to });
	currentPoint = to;
	currentContour->ExpandBoundingBox(currentPoint);
	return 0;
}

int GlyphDecomposer::LineTo(const XGLVertex& to) {
	//xprintf(" LineTo: %0.4f %0.4f\n", to.x, to.y);

	if (IsEqual(to, firstPoint)) {
		//xprintf(" LineTo: %0.4f %0.4f is coincident with Contour start, ignoring\n");
		return 0;
	}

	glyphOutline[contourIdx].v.push_back({ to });
	currentPoint = to;
	currentContour->ExpandBoundingBox(currentPoint);
	return 0;
}

int GlyphDecomposer::ConicTo(const XGLVertex& control, const XGLVertex& to) {
	//xprintf("ConicTo: %0.4f %0.4f - %0.4f %0.4f\n", control.x, control.y, to.x, to.y);
	if (drawCurves) {
		EvaluateQuadraticBezier(currentPoint, control, to);
	}
	else {
		glyphOutline[contourIdx].v.push_back({ control });
		currentPoint = control;
		currentContour->ExpandBoundingBox(currentPoint);

		if (IsEqual(to, firstPoint))
			return 0;

		glyphOutline[contourIdx].v.push_back({ to });
	}
	currentPoint = to;
	currentContour->ExpandBoundingBox(currentPoint);
	return 0;
}

int GlyphDecomposer::CubicTo(const XGLVertex& control1, const XGLVertex& control2, const XGLVertex& to) {
	//xprintf(" CubeTo: %0.4f %0.4f - %0.4f %0.4f - %0.4f %0.4f\n", control1.x, control1.y, control2.x, control2.y, to.x, to.y);
	if (drawCurves) {
		EvaluateCubicBezier(currentPoint, control1, control2, to);
	}
	else {
		glyphOutline[contourIdx].v.push_back({ control1 });
		glyphOutline[contourIdx].v.push_back({ control2 });
		currentPoint = control2;
		currentContour->ExpandBoundingBox(currentPoint);

		if (IsEqual(to, firstPoint))
			return 0;

		glyphOutline[contourIdx].v.push_back({ to });
	}
	currentPoint = to;
	currentContour->ExpandBoundingBox(currentPoint);
	return 0;
}

const XGLVertex GlyphDecomposer::Interpolate(const XGLVertex& p1, const XGLVertex& p2, float percent) {
	XGLVertex v;
	REAL diff;

	diff = p2.x - p1.x;
	v.x = p1.x + (diff * percent);
	diff = p2.y - p1.y;
	v.y = p1.y + (diff * percent);

	return v;
}

void GlyphDecomposer::EvaluateQuadraticBezier(const XGLVertex& p0, const XGLVertex& p1, const XGLVertex& p2) {
	REAL interpolant;
	XGLVertex i0, i1, out;

	for (interpolant = interpolationFactor; interpolant < 1.0f; interpolant += interpolationFactor) {
		i0 = Interpolate(p0, p1, interpolant);
		i1 = Interpolate(p1, p2, interpolant);
		out = Interpolate(i0, i1, interpolant);
		glyphOutline[contourIdx].v.push_back({ out });
	}
	if (IsEqual(p2, firstPoint)) {
		//xprintf("Final point of Quadratic Bezier is coincident with contour start, ignoring\n");
		return;
	}
	glyphOutline[contourIdx].v.push_back({ p2 });
}

void GlyphDecomposer::EvaluateCubicBezier(const XGLVertex& p0, const XGLVertex& p1, const XGLVertex& p2, const XGLVertex& p3) {
	REAL interpolant;
	XGLVertex i0, i1, i2, m0, m1, out;

	for (interpolant = interpolationFactor; interpolant < 1.0f; interpolant += interpolationFactor) {
		i0 = Interpolate(p0, p1, interpolant);
		i1 = Interpolate(p1, p2, interpolant);
		i2 = Interpolate(p2, p3, interpolant);

		m0 = Interpolate(i0, i1, interpolant);
		m1 = Interpolate(i1, i2, interpolant);

		out = Interpolate(m0, m1, interpolant);
		glyphOutline[contourIdx].v.push_back({ out });
	}
	if (IsEqual(p3, firstPoint)) {
		//xprintf("Final point of Cubic Bezier is coincident with contour start, ignoring\n");
		return;
	}
	glyphOutline[contourIdx].v.push_back({ p3 });
}
