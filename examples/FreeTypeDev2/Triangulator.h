#ifndef TRIANGULATOR_H
#define TRIANGULATOR_H

#include <string>
#include <fstream>
#include <sstream>
#include <vector>
#include "xgl.h"
#include "XGLFreeType.h"

// triangle.h wants this
#ifndef REAL
#define REAL double
#endif
extern "C" {
#define ANSI_DECLARATORS
#include "triangle.h"
};

extern int numPoints;
extern int num2draw;

enum polyParseState {
	GET_POINTS_HEADER,
	GET_POINTS,
	GET_SEGMENTS_HEADER,
	GET_SEGMENTS,
	GET_HOLES_HEADER,
	GET_HOLES,
	GET_REGION_HEADER,
	GET_REGIONS
};

class Triangulator : public triangulateio, public XGLShape {
public:
	Triangulator();

	void Init(triangulateio& in);
	void Draw();

	void Convert(FT::GlyphOutline&, triangulateio&, XGLVertex&, REAL);
	void RenderTriangles(triangulateio& in);
	void RenderSegments(triangulateio& t);

	void Dump(triangulateio& in);
	void SetDrawCount(GLsizei count);

private:
	GLuint drawMode = GL_LINES; // GL_LINES or GL_TRIANGES (for filling in)
	GLsizei drawCount;
};


#endif