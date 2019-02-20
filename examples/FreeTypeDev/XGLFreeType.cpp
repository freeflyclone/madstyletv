#include "XGLFreeType.h"

using namespace FT;

int GlyphDecomposer::MoveTo(const FT_Vector& to) {
	xprintf(" MoveTo: %d %d\n", to.x, to.y);
	firstPoint = to;

	// if the current Contour has FT_Vector data, this MoveTo is new Contour
	if (glyphOutline[contourIdx].size()) {
		contourIdx++;
		glyphOutline.push_back(*(new Contour()));
	}

	glyphOutline[contourIdx].push_back(to);
	currentPoint = to;
	return 0;
}

int GlyphDecomposer::LineTo(const FT_Vector& to) {
	xprintf(" LineTo: %d %d\n", to.x, to.y);

	if (IsEqual(to, firstPoint))
		return 0;

	glyphOutline[contourIdx].push_back(to);
	currentPoint = to;
	return 0;
}

int GlyphDecomposer::ConicTo(const FT_Vector& control, const FT_Vector& to) {
	xprintf("ConicTo: %d %d - %d %d\n", control.x, control.y, to.x, to.y);
	glyphOutline[contourIdx].push_back(control);
	currentPoint = control;

	if (IsEqual(to, firstPoint))
		return 0;

	glyphOutline[contourIdx].push_back(to);
	currentPoint = to;
	return 0;
}

int GlyphDecomposer::CubicTo(const FT_Vector& control1, const FT_Vector& control2, const FT_Vector& to) {
	xprintf(" CubeTo: %d %d - %d %d - %d %d\n", control1.x, control1.y, control2.x, control2.y, to.x, to.y);
	glyphOutline[contourIdx].push_back(control1);
	glyphOutline[contourIdx].push_back(control2);
	currentPoint = control2;

	if (IsEqual(to, firstPoint))
		return 0;

	glyphOutline[contourIdx].push_back(to);
	currentPoint = to;
	return 0;
}
