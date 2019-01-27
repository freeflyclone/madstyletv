/**************************************************************
** FreeTypeDevBuildScene.cpp
**
** Demonstrate drawing FreeType outlines using 
** FT_Outline_Decompose. Found the EvaluateXXXBezier() methods
** on StackOverFlow.
**************************************************************/
#include "ExampleXGL.h"

#include FT_OUTLINE_H

// triangle.h wants this
#ifndef REAL
#define REAL double
#endif
extern "C" {
	#define ANSI_DECLARATORS
	#define TRILIBRARY
    #define REDUCED
    #define CDT_ONLY
	#include "triangle.h"
};

#ifdef FONT_NAME
#undef FONT_NAME
#endif

#define FONT_NAME "C:/windows/fonts/times.ttf"

class Triangulator : public XGLShape, public triangulateio {
public:
	Triangulator() {
		struct triangulateio in, mid;

		/* Define input points. */

		in.numberofpoints = 4;
		in.numberofpointattributes = 1;
		in.pointlist = (REAL *)malloc(in.numberofpoints * 2 * sizeof(REAL));
		in.pointlist[0] = 0.0;
		in.pointlist[1] = 0.0;
		in.pointlist[2] = 1.0;
		in.pointlist[3] = 0.0;
		in.pointlist[4] = 1.0;
		in.pointlist[5] = 10.0;
		in.pointlist[6] = 0.0;
		in.pointlist[7] = 10.0;
		in.pointattributelist = (REAL *)malloc(in.numberofpoints *
			in.numberofpointattributes *
			sizeof(REAL));
		in.pointattributelist[0] = 0.0;
		in.pointattributelist[1] = 1.0;
		in.pointattributelist[2] = 11.0;
		in.pointattributelist[3] = 10.0;
		in.pointmarkerlist = (int *)malloc(in.numberofpoints * sizeof(int));
		in.pointmarkerlist[0] = 0;
		in.pointmarkerlist[1] = 2;
		in.pointmarkerlist[2] = 0;
		in.pointmarkerlist[3] = 0;

		in.numberofsegments = 0;
		in.numberofholes = 0;
		in.numberofregions = 1;
		in.regionlist = (REAL *)malloc(in.numberofregions * 4 * sizeof(REAL));
		in.regionlist[0] = 0.5;
		in.regionlist[1] = 5.0;
		in.regionlist[2] = 7.0;            /* Regional attribute (for whole mesh). */
		in.regionlist[3] = 0.1;          /* Area constraint that will not be used. */

		printf("Input point set:\n\n");
		report(&in, 1, 1, 1, 1, 1, 1);

		/* Make necessary initializations so that Triangle can return a */
		/*   triangulation in `mid' and a voronoi diagram in `vorout'.  */

		mid.pointlist = (REAL *)NULL;            /* Not needed if -N switch used. */
		/* Not needed if -N switch used or number of point attributes is zero: */
		mid.pointattributelist = (REAL *)NULL;
		mid.pointmarkerlist = (int *)NULL; /* Not needed if -N or -B switch used. */
		mid.trianglelist = (int *)NULL;          /* Not needed if -E switch used. */
		/* Not needed if -E switch used or number of triangle attributes is zero: */
		mid.triangleattributelist = (REAL *)NULL;
		mid.neighborlist = (int *)NULL;         /* Needed only if -n switch used. */
		/* Needed only if segments are output (-p or -c) and -P not used: */
		mid.segmentlist = (int *)NULL;
		/* Needed only if segments are output (-p or -c) and -P and -B not used: */
		mid.segmentmarkerlist = (int *)NULL;
		mid.edgelist = (int *)NULL;             /* Needed only if -e switch used. */
		mid.edgemarkerlist = (int *)NULL;   /* Needed if -e used and -B not used. */

		/* Triangulate the points.  Switches are chosen to read and write a  */
		/*   PSLG (p), preserve the convex hull (c), number everything from  */
		/*   zero (z), assign a regional attribute to each element (A), and  */
		/*   produce an edge list (e), a Voronoi diagram (v), and a triangle */
		/*   neighbor list (n).                                              */

		triangulate("pczAen", &in, &mid, nullptr);
	}
	void report(struct triangulateio *io,int markers, int reporttriangles, int reportneighbors, int reportsegments, int reportedges, int reportnorms) {
	  int i, j;

	  for (i = 0; i < io->numberofpoints; i++) {
		xprintf("Point %4d:", i);
		for (j = 0; j < 2; j++) {
		  xprintf("  %.6g", io->pointlist[i * 2 + j]);
		}
		if (io->numberofpointattributes > 0) {
		  xprintf("   attributes");
		}
		for (j = 0; j < io->numberofpointattributes; j++) {
		  xprintf("  %.6g",
				 io->pointattributelist[i * io->numberofpointattributes + j]);
		}
		if (markers) {
		  xprintf("   marker %d\n", io->pointmarkerlist[i]);
		} else {
		  xprintf("\n");
		}
	  }
	  xprintf("\n");

	  if (reporttriangles || reportneighbors) {
		for (i = 0; i < io->numberoftriangles; i++) {
		  if (reporttriangles) {
			xprintf("Triangle %4d points:", i);
			for (j = 0; j < io->numberofcorners; j++) {
			  xprintf("  %4d", io->trianglelist[i * io->numberofcorners + j]);
			}
			if (io->numberoftriangleattributes > 0) {
			  xprintf("   attributes");
			}
			for (j = 0; j < io->numberoftriangleattributes; j++) {
			  xprintf("  %.6g", io->triangleattributelist[i *
											 io->numberoftriangleattributes + j]);
			}
			xprintf("\n");
		  }
		  if (reportneighbors) {
			xprintf("Triangle %4d neighbors:", i);
			for (j = 0; j < 3; j++) {
			  xprintf("  %4d", io->neighborlist[i * 3 + j]);
			}
			xprintf("\n");
		  }
		}
		xprintf("\n");
	  }

	  if (reportsegments) {
		for (i = 0; i < io->numberofsegments; i++) {
		  xprintf("Segment %4d points:", i);
		  for (j = 0; j < 2; j++) {
			xprintf("  %4d", io->segmentlist[i * 2 + j]);
		  }
		  if (markers) {
			xprintf("   marker %d\n", io->segmentmarkerlist[i]);
		  } else {
			xprintf("\n");
		  }
		}
		xprintf("\n");
	  }

	  if (reportedges) {
		for (i = 0; i < io->numberofedges; i++) {
		  xprintf("Edge %4d points:", i);
		  for (j = 0; j < 2; j++) {
			xprintf("  %4d", io->edgelist[i * 2 + j]);
		  }
		  if (reportnorms && (io->edgelist[i * 2 + 1] == -1)) {
			for (j = 0; j < 2; j++) {
			  xprintf("  %.6g", io->normlist[i * 2 + j]);
			}
		  }
		  if (markers) {
			xprintf("   marker %d\n", io->edgemarkerlist[i]);
		  } else {
			xprintf("\n");
		  }
		}
		xprintf("\n");
	  }
	}

};

class XGLFreeType : public XGLShape {
private:
	// A class to map static C callback functions called by 
	// FT_Outline_Decompose() to XGLFreeType instance methods.
	class FreeTypeDecomposer : public FT_Outline_Funcs {
	public:
		static int _MoveToFunc(const FT_Vector* to, void *pCtx) {
			return ((XGLFreeType*)pCtx)->MoveTo(to);
		}
		static int _LineToFunc(const FT_Vector* to, void *pCtx) {
			return ((XGLFreeType*)pCtx)->LineTo(to);
		}
		static int _ConicToFunc(const FT_Vector*	control, const FT_Vector* to, void *pCtx) {
			return ((XGLFreeType*)pCtx)->ConicTo(control, to);
		}
		static int _CubicToFunc(const FT_Vector*	control1, const FT_Vector*	control2, const FT_Vector* to, void *pCtx) {
			return ((XGLFreeType*)pCtx)->CubicTo(control1, control2, to);
		}

		FreeTypeDecomposer() {
			move_to = _MoveToFunc;
			line_to = _LineToFunc;
			conic_to = _ConicToFunc;
			cubic_to = _CubicToFunc;
			shift = 0;
			delta = 0;
		}
	};
public:
	typedef std::map<FT_ULong, FT_UInt> CharMap;
	typedef std::vector<GLsizei> ContourOffsets;

	XGLFreeType(std::string t) : textToRender(t) {
		FT_UInt gindex = 0;
		FT_ULong charcode = 0;

		if (FT_Init_FreeType(&ft))
			throwXGLException("Init of FreeType failed");

		if (FT_New_Face(ft, FONT_NAME, 0, &face))
			throwXGLException("FT_New_Face() failed " FONT_NAME);

		if (FT_Select_Charmap(face, ft_encoding_unicode))
			throwXGLException("FT_Select_Charmap(UNICODE) failed.");

		// scale the rendering to some ridiculous size.
		// This is required, else the outline points will all be zero.
		// NOTE: the first two #s are 26.6 fixed point, so 256 is actually 4.0
		//       The second pair are DPI numbers.
		//       Both pairs ought to be fairly large, so that
		//		 FreeType's internal math doesn't bodge the precision of the outline.
		FT_Set_Char_Size(face, 256, 0, 512, 0);

		FT_GlyphSlot g = face->glyph;

		// build an XGLCharMap of the entire set of glyphs for this font.
		// (this could be huge for Chinese fonts)
		for (charcode = FT_Get_First_Char(face, &gindex); gindex; charcode = FT_Get_Next_Char(face, charcode, &gindex))
			charMap.emplace(charcode, gindex);

		const int numGlyphs = (const int)(charMap.size());

		currentPoint = { 0, 0 };
		advance = { 0, 0 };

		for (auto c : textToRender) {
			gindex = charMap[c];

			FT_Load_Glyph(face, gindex, FT_LOAD_FORCE_AUTOHINT | FT_LOAD_TARGET_NORMAL);
			if (true)
			{
				drawCurves = false;
				FT_Outline_Decompose(&g->outline, &fdc, this);
				contourOffsets.push_back((int)v.size());
			}
			if (true)
			{
				drawCurves = true;
				FT_Outline_Decompose(&g->outline, &fdc, this);
				contourOffsets.push_back((int)v.size());
			}
			advance.x += g->advance.x;
			advance.y += g->advance.y;
		}
	};

	~XGLFreeType() {
		FT_Done_Face(face);
		FT_Done_FreeType(ft);
	};

	void Draw() {
		if (v.size()) {
			int i = 0;
			for (auto c : contourOffsets) {
				glDrawArrays(GL_LINE_LOOP, i, (GLsizei)(c-i));
				GL_CHECK("glDrawArrays() failed");
				i = c + 1;
			}
		}
	}

	FT_Vector Advance(const FT_Vector* vector) {
		return{ advance.x + vector->x, advance.y + vector->y };
	}

	int MoveTo(const FT_Vector* to) {
		// mark our progress along the outline
		currentPoint = *to;

		// if this isn't the very first vertex...
		if (v.size() > 0) {
			//...we've seen vertices, is this the very first contour?...
			if (contourOffsets.empty())
				contourOffsets.push_back((int)v.size());
			// not first contour ever, ensure it isn't the first contour of new glyph
			else if (v.size() > contourOffsets.back())
				contourOffsets.push_back((int)v.size());
		}

		// add the first point of the new contour
		v.push_back({ { Advance(to).x / scaleFactor, Advance(to).y / scaleFactor, 0 }, {}, {}, pointsColor });

		return 0;
	}

	int LineTo(const FT_Vector* to) {
		v.push_back({ { Advance(to).x / scaleFactor, Advance(to).y / scaleFactor, 0 }, {}, {}, pointsColor });
		currentPoint = *to;
		return 0;
	}

	int ConicTo(const FT_Vector* control, const FT_Vector* to) {
		if (drawCurves)
			EvaluateQuadraticBezier(Advance(&currentPoint), Advance(control), Advance(to));
		else
			v.push_back({ { Advance(to).x / scaleFactor, Advance(to).y / scaleFactor, 0 }, {}, {}, pointsColor });
		currentPoint = *to;
		return 0;
	}

	int CubicTo(const FT_Vector* control1, const FT_Vector* control2, const FT_Vector* to) {
		if (drawCurves)
			EvaluateCubicBezier(Advance(&currentPoint), Advance(control1), Advance(control2), Advance(to));
		else
			v.push_back({ { Advance(to).x / scaleFactor, Advance(to).y / scaleFactor, 0 }, {}, {}, pointsColor });
		currentPoint = *to;
		return 0;
	}

	float GetInterpolatedPoint(float n1, float n2, float percent) {
		float diff = n2 - n1;
		return n1 + (diff * percent);
	}

	void EvaluateQuadraticBezier(FT_Vector p0, FT_Vector p1, FT_Vector p2) {
		float xa, xb, ya, yb;
		float x, y;
		float interpolant;

		for (interpolant = 0.0f; interpolant < 1.0f; interpolant += interpolationFactor) {
			xa = GetInterpolatedPoint((float)p0.x, (float)p1.x, interpolant);
			ya = GetInterpolatedPoint((float)p0.y, (float)p1.y, interpolant);

			xb = GetInterpolatedPoint((float)p1.x, (float)p2.x, interpolant);
			yb = GetInterpolatedPoint((float)p1.y, (float)p2.y, interpolant);

			x = GetInterpolatedPoint(xa, xb, interpolant);
			y = GetInterpolatedPoint(ya, yb, interpolant);

			v.push_back({ { x/scaleFactor, y/scaleFactor, 0 }, {}, {}, curvesColor });
		}
		v.push_back({ { p2.x / scaleFactor, p2.y / scaleFactor, 0 }, {}, {}, curvesColor });
		v.push_back({ { p2.x / scaleFactor, p2.y / scaleFactor, 0 }, {}, {}, controlColor });
		v.push_back({ { p1.x / scaleFactor, p1.y / scaleFactor, 0 }, {}, {}, controlColor });
		v.push_back({ { p0.x / scaleFactor, p0.y / scaleFactor, 0 }, {}, {}, controlColor });
		v.push_back({ { p1.x / scaleFactor, p1.y / scaleFactor, 0 }, {}, {}, controlColor });
		v.push_back({ { p2.x / scaleFactor, p2.y / scaleFactor, 0 }, {}, {}, controlColor });
		v.push_back({ { p2.x / scaleFactor, p2.y / scaleFactor, 0 }, {}, {}, curvesColor });
	}

	void EvaluateCubicBezier(FT_Vector p0, FT_Vector p1, FT_Vector p2, FT_Vector p3) {
		float xa, xb, xc, ya, yb, yc;
		float xm, xn, ym, yn;
		float x, y;
		float interpolant;

		for (interpolant = 0.0f; interpolant < 1.0f; interpolant += interpolationFactor) {
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

			v.push_back({ { x / scaleFactor, y / scaleFactor, 0 }, {}, {}, curvesColor });
		}
		v.push_back({ { p3.x / scaleFactor, p3.y / scaleFactor, 0 }, {}, {}, curvesColor });
	}

	bool drawCurves;
	XGLColor pointsColor = XGLColors::yellow;
	XGLColor curvesColor = XGLColors::cyan;
	XGLColor controlColor = XGLColors::magenta;

	std::string textToRender;
	FreeTypeDecomposer fdc;
	ContourOffsets contourOffsets;
	FT_Library ft;
	FT_Face face;
	FT_GlyphSlot g;
	CharMap charMap;

	FT_Vector currentPoint;
	FT_Vector advance;

	float scaleFactor = 200.0f;
	float interpolationFactor = 0.05f;
};

void ExampleXGL::BuildScene() {
	XGLFreeType *shape;
	Triangulator *t;

	AddShape("shaders/000-simple", [&](){ shape = new XGLFreeType(config.WideToBytes(config.Find(L"FreeTypeText")->AsString())); return shape; });
	glm::mat4 translate = glm::translate(glm::mat4(), glm::vec3(0, 0, 1.0));
	shape->model = translate;

	AddShape("shaders/000-simple", [&](){ t = new Triangulator(); return t; });
	translate = glm::translate(glm::mat4(), glm::vec3(5, 10, 1.0));
	t->model = translate;
}
