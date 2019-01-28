/**************************************************************
** FreeTypeDevBuildScene.cpp
**
** Demonstrate drawing FreeType outlines using 
** FT_Outline_Decompose. Found the EvaluateXXXBezier() methods
** on StackOverFlow.
**************************************************************/
#include "ExampleXGL.h"
#include <string>

#include FT_OUTLINE_H

// triangle.h wants this
#ifndef REAL
#define REAL double
#endif
extern "C" {
	#define ANSI_DECLARATORS
	#define TRILIBRARY
    //#define REDUCED
    //#define CDT_ONLY
	#include "triangle.h"
};

#ifdef FONT_NAME
#undef FONT_NAME
#endif

#define FONT_NAME "C:/windows/fonts/times.ttf"

class Triangulator : public XGLShape, public triangulateio {
public:
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

	void ReadPolyFile(std::string fileName, triangulateio* t) {
		std::ifstream file(fileName);
		std::string line;
		polyParseState ps = GET_POINTS_HEADER;
		int numberPointMarkers = 0;
		int numberSegmentMarkers = 0;
		double x, y;

		while (std::getline(file, line)) {
			std::stringstream ss(line);
			std::vector<std::string> tokens;
			std::string token;
			int index, p1, p2, p3;

			// we know our input is tokenized with whitespace, so the following works.
			while (ss >> token)
				tokens.push_back(token);

			switch (ps) {
				case GET_POINTS_HEADER:
					t->numberofpoints = std::stoi(tokens[0]);
					t->numberofpointattributes = std::stoi(tokens[2]);
					numberPointMarkers = std::stoi(tokens[3]);
					t->pointlist = (REAL *)malloc(t->numberofpoints * 2 * sizeof(REAL));
					t->pointattributelist = (REAL *)malloc(t->numberofpoints * t->numberofpointattributes *	sizeof(REAL));
					ps = GET_POINTS;
					break;

				case GET_POINTS:
					index = std::stoi(tokens[0]) - 1;
					x = std::stod(tokens[1]);
					y = std::stod(tokens[2]);
					t->pointlist[index * 2] = x;
					t->pointlist[index * 2 + 1] = y;

					for (int i = 0; i < t->numberofpointattributes; i++)
						t->pointattributelist[index * t->numberofpointattributes + i] = std::stod(tokens[3]);

					if ((1+index) == t->numberofpoints)
						ps = GET_SEGMENTS_HEADER;
					break;

				case GET_SEGMENTS_HEADER:
					t->numberofsegments = std::stoi(tokens[0]);
					numberSegmentMarkers = std::stoi(tokens[1]);
					t->segmentlist = (int*)malloc(t->numberofsegments * 2 * sizeof(int));
					ps = GET_SEGMENTS;
					break;

				case GET_SEGMENTS:
					index = std::stoi(tokens[0]) - 1;
					p1 = std::stoi(tokens[1]);
					p2 = std::stoi(tokens[2]);
					t->segmentlist[index * 2] = p1;
					t->segmentlist[index * 2 + 1] = p2;
					if ((1 + index) == t->numberofsegments)
						ps = GET_HOLES_HEADER;
					break;

				case GET_HOLES_HEADER:
					t->numberofholes = std::stoi(tokens[0]);
					t->holelist = (REAL*)malloc(2 * t->numberofcorners * sizeof(REAL));
					ps = GET_HOLES;
					break;

				case GET_HOLES:
					index = std::stoi(tokens[0]) - 1;
					x = std::stod(tokens[1]);
					y = std::stod(tokens[2]);
					t->holelist[index * 2] = x;
					t->holelist[index * 2 + 1] = y;
					if ((1 + index) == t->numberofholes)
						ps = GET_REGION_HEADER;
					break;
					
				default:
					xprintf("The line is: %s\n", line.c_str());
					break;
			}
		}

		/*
		t->numberofpoints = 4;
		t->numberofpointattributes = 0;
		t->pointlist = (REAL *)malloc(t->numberofpoints * 2 * sizeof(REAL));
		t->pointlist[0] = 0.0;
		t->pointlist[1] = 0.0;
		t->pointlist[2] = 1.0;
		t->pointlist[3] = 0.0;
		t->pointlist[4] = 1.0;
		t->pointlist[5] = 10.0;
		t->pointlist[6] = 0.0;
		t->pointlist[7] = 10.0;
		t->pointattributelist = (REAL *)malloc(t->numberofpoints * t->numberofpointattributes *	sizeof(REAL));
		t->pointattributelist[0] = 0.0;
		t->pointattributelist[1] = 0.0;// 1.0;
		t->pointattributelist[2] = 0.0;// 11.0;
		t->pointattributelist[3] = 0.0;// 10.0;

		/*
		t->pointmarkerlist = (int *)malloc(t->numberofpoints * sizeof(int));
		t->pointmarkerlist[0] = 0;
		t->pointmarkerlist[1] = 2;
		t->pointmarkerlist[2] = 0;
		t->pointmarkerlist[3] = 0;

		t->numberofsegments = 0;
		t->numberofholes = 0;

		t->numberofregions = 1;
		t->regionlist = (REAL *)malloc(t->numberofregions * 4 * sizeof(REAL));
		t->regionlist[0] = 0.5;
		t->regionlist[1] = 5.0;
		t->regionlist[2] = 7.0;            // Regional attribute (for whole mesh).
		t->regionlist[3] = 0.1;          // Area constraint that will not be used.
		*/
	}

	Triangulator() {
		struct triangulateio in{ 0 }, mid{ 0 };

		ReadPolyFile("../assets/a.poly", &in);
		/* Define input points. */

		//xprintf("Input point set:\n\n");
		//report(&in, 1, 1, 1, 1, 1, 1);

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

		triangulate("pcz", &in, &mid, nullptr);

		xprintf("triangulate() output: (numberofcorners: %d)\n", mid.numberofcorners);
		report(&mid, 1, 1, 0, 0, 0, 0);

		//xyzzy
		// during dev, use GL_LINES to visualize, so push 2 vertices per outline point,
		// the point we're on, plus the one that comes after. Handle wraparound
		// with the usual modulo trick.
		for (int i = 0; i < mid.numberoftriangles; i++) {
			for (int j = 0; j < mid.numberofcorners; j++) {
				int idx = mid.trianglelist[i * mid.numberofcorners + j];
				// modulo trick: get the next (possibly wrapped) vertex of *this* triangle
				int idxNext = mid.trianglelist[(i*mid.numberofcorners + ((j+1) % mid.numberofcorners))];

				// first point of the line segment of this triangle's edge
				REAL x = mid.pointlist[idx * 2];
				REAL y = mid.pointlist[idx * 2 + 1];

				// and the second point this triangle's edge
				REAL x2 = mid.pointlist[idxNext * 2];
				REAL y2 = mid.pointlist[idxNext * 2 + 1];

				// put them in the Vertex buffer for rendering.
				v.push_back({ { x, y, 1 }, {}, {}, { XGLColors::white } });
				v.push_back({ { x2, y2, 1 }, {}, {}, { XGLColors::white } });
			}
		}
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

	void Draw() {
		if (v.size()) {
			glDrawArrays(GL_LINES, 0, (GLsizei)(v.size()));
			GL_CHECK("glDrawArrays() failed");
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
	glm::mat4 scale = glm::scale(glm::mat4(), glm::vec3(4, 4, 4));
	t->model = translate * scale;
}
