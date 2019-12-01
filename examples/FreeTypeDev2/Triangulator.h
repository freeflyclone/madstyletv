#ifndef TRIANGULATOR_H
#define TRIANGULATOR_H

#include <string>
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

class Triangulator : public triangulateio, public XGLShape {
public:
	Triangulator();

	void Init(triangulateio& in);
	void Free(triangulateio& in, bool);

	void Convert(FT::GlyphOutline&, XGLVertex&);
	void RenderTriangles(triangulateio& in);

	static REAL ScaleFactor() { return 3276.80f; }

private:
	const REAL scaleFactor{ 3276.80 };
};


#endif