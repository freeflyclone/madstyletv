#include "XGLFreeType.h"

using namespace FT;

int GlyphDecomposer::MoveTo(const XGLVertex& to) {
	xprintf(" MoveTo: %0.4f %0.4f\n", to.x, to.y);
	firstPoint = to;

	// if the current Contour has XGLVertex data, this MoveTo is a new Contour
	if (glyphOutline[contourIdx].size()) {
		contourIdx++;
		glyphOutline.push_back(*(new Contour()));
	}

	glyphOutline[contourIdx].push_back({ to });
	currentPoint = to;
	return 0;
}

int GlyphDecomposer::LineTo(const XGLVertex& to) {
	xprintf(" LineTo: %0.4f %0.4f\n", to.x, to.y);

	if (IsEqual(to, firstPoint)) {
		xprintf(" LineTo: %0.4f %0.4f is coincident with Contour start, ignoring\n");
		return 0;
	}

	glyphOutline[contourIdx].push_back({ to });
	currentPoint = to;
	return 0;
}

int GlyphDecomposer::ConicTo(const XGLVertex& control, const XGLVertex& to) {
	xprintf("ConicTo: %0.4f %0.4f - %0.4f %0.4f\n", control.x, control.y, to.x, to.y);
	if (drawCurves) {
		EvaluateQuadraticBezier(currentPoint, control, to);
	}
	else {
		glyphOutline[contourIdx].push_back({ control });
		currentPoint = control;

		if (IsEqual(to, firstPoint))
			return 0;

		glyphOutline[contourIdx].push_back({ to });
	}
	currentPoint = to;
	return 0;
}

int GlyphDecomposer::CubicTo(const XGLVertex& control1, const XGLVertex& control2, const XGLVertex& to) {
	xprintf(" CubeTo: %0.4f %0.4f - %0.4f %0.4f - %0.4f %0.4f\n", control1.x, control1.y, control2.x, control2.y, to.x, to.y);
	if (drawCurves) {
		EvaluateCubicBezier(currentPoint, control1, control2, to);
	}
	else {
		glyphOutline[contourIdx].push_back({ control1 });
		glyphOutline[contourIdx].push_back({ control2 });
		currentPoint = control2;

		if (IsEqual(to, firstPoint))
			return 0;

		glyphOutline[contourIdx].push_back({ to });
	}
	currentPoint = to;
	return 0;
}

const XGLVertex& GlyphDecomposer::Interpolate(const XGLVertex& p1, const XGLVertex& p2, float percent) {
	XGLVertex v;
	float diff;

	diff = p2.x - p1.x;
	v.x = p1.x + (diff * percent);
	diff = p2.y - p1.y;
	v.y = p1.y + (diff * percent);

	return v;
}
void GlyphDecomposer::EvaluateQuadraticBezier(const XGLVertex& p0, const XGLVertex& p1, const XGLVertex& p2) {
	float interpolant;
	XGLVertex i0, i1, out;

	for (interpolant = interpolationFactor; interpolant < 1.0f; interpolant += interpolationFactor) {
		i0 = Interpolate(p0, p1, interpolant);
		i1 = Interpolate(p1, p2, interpolant);
		out = Interpolate(i0, i1, interpolant);
		glyphOutline[contourIdx].push_back({ out });
	}
	if (IsEqual(p2, firstPoint)) {
		xprintf("Final point of Quadratic Bezier is coincident with contour start, ignoring\n");
		return;
	}
	glyphOutline[contourIdx].push_back({ p2 });
}

void GlyphDecomposer::EvaluateCubicBezier(const XGLVertex& p0, const XGLVertex& p1, const XGLVertex& p2, const XGLVertex& p3) {
	float interpolant;
	XGLVertex i0, i1, i2, m0, m1, out;

	for (interpolant = interpolationFactor; interpolant < 1.0f; interpolant += interpolationFactor) {
		i0 = Interpolate(p0, p1, interpolant);
		i1 = Interpolate(p1, p2, interpolant);
		i2 = Interpolate(p2, p3, interpolant);

		m0 = Interpolate(i0, i1, interpolant);
		m1 = Interpolate(i1, i2, interpolant);

		out = Interpolate(m0, m1, interpolant);
		glyphOutline[contourIdx].push_back({ out });
	}
	if (IsEqual(p3, firstPoint)) {
		xprintf("Final point of Cubic Bezier is coincident with contour start, ignoring\n");
		return;
	}
	glyphOutline[contourIdx].push_back({ p3 });
}
/*
function IsClockwise(feature)
{
	if (feature.geometry == null)
		return -1;

	var vertices = feature.geometry.getVertices();
	var area = 0;

	for (var i = 0; i < (vertices.length); i++) {
		j = (i + 1) % vertices.length;

		area += vertices[i].x * vertices[j].y;
		area -= vertices[j].x * vertices[i].y;
		// console.log(area);
	}

	return (area < 0);
}
*/