/**************************************************************
** FreeTypeDevBuildScene.cpp
**
** Demonstrate drawing FreeType outlines using 
** FT_Outline_Decompose. Found the EvaluateXXXBezier() methods
** on StackOverFlow.
**************************************************************/
#include "ExampleXGL.h"
#include <string>

#include "Triangulator.h"

#include FT_OUTLINE_H


#ifdef FONT_NAME
#undef FONT_NAME
#endif

#define FONT_NAME "C:/windows/fonts/arial.ttf"


int numPoints = 0;
int num2draw;

int numPoints2 = 0;
int num2draw2;

class XGLFreeType : public XGLShape {
public:
	typedef std::map<FT_ULong, FT_UInt> CharMap;
	typedef std::vector<GLsizei> ContourEndPoints;

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

	// use the "Triangle" package from http://www.cs.cmu.edu/~quake/triangle.html
	class Triangulator {
	public:
		void ReadPolyFile(std::string fileName, triangulateio& in) {
			std::ifstream file(fileName);
			std::string line;
			polyParseState ps = GET_POINTS_HEADER;
			int numberPointMarkers = 0;
			int numberSegmentMarkers = 0;
			double x, y;
			int lineNum{ 0 };

			while (std::getline(file, line)) {
				std::stringstream ss(line);
				std::vector<std::string> tokens;
				std::string token;
				int index, p1, p2, p3;

				// we know our input is tokenized with whitespace, so the following works.
				while (ss >> token)
					tokens.push_back(token);

				lineNum++;
				switch (ps) {
				case GET_POINTS_HEADER:
					in.numberofpoints = std::stoi(tokens[0]);
					in.numberofpointattributes = std::stoi(tokens[2]);
					numberPointMarkers = std::stoi(tokens[3]);
					in.pointlist = (REAL *)malloc(in.numberofpoints * 2 * sizeof(REAL));
					in.pointattributelist = (REAL *)malloc(in.numberofpoints * in.numberofpointattributes *	sizeof(REAL));
					ps = GET_POINTS;
					break;

				case GET_POINTS:
					index = std::stoi(tokens[0]) - 1;
					x = std::stod(tokens[1]);
					y = std::stod(tokens[2]);
					in.pointlist[index * 2] = x;
					in.pointlist[index * 2 + 1] = y;

					for (int i = 0; i < in.numberofpointattributes; i++)
						in.pointattributelist[index * in.numberofpointattributes + i] = std::stod(tokens[3 + i]);

					if ((1 + index) == in.numberofpoints) {
						ps = GET_SEGMENTS_HEADER;
					}
					break;

				case GET_SEGMENTS_HEADER:
					in.numberofsegments = std::stoi(tokens[0]);
					numberSegmentMarkers = std::stoi(tokens[1]);
					in.segmentlist = (int*)malloc(in.numberofsegments * 2 * sizeof(int));
					ps = GET_SEGMENTS;
					break;

				case GET_SEGMENTS:
					index = std::stoi(tokens[0]) - 1;
					p1 = std::stoi(tokens[1]);
					p2 = std::stoi(tokens[2]);
					// we're using "start from zero" triangulation, input poly file starts from one,
					// so compensate by subtracting 1 from all input point indexes.
					in.segmentlist[index * 2] = p1 - 1;
					in.segmentlist[index * 2 + 1] = p2 - 1;
					if ((1 + index) == in.numberofsegments)
						ps = GET_HOLES_HEADER;
					break;

				case GET_HOLES_HEADER:
					in.numberofholes = std::stoi(tokens[0]);
					in.holelist = (REAL*)malloc(2 * in.numberofholes * sizeof(REAL));
					ps = GET_HOLES;
					break;

				case GET_HOLES:
					index = std::stoi(tokens[0]) - 1;
					x = std::stod(tokens[1]);
					y = std::stod(tokens[2]);
					in.holelist[index * 2] = x;
					in.holelist[index * 2 + 1] = y;
					if ((1 + index) == in.numberofholes)
						ps = GET_REGION_HEADER;
					break;

				default:
					xprintf("The line is: %s\n", line.c_str());
					break;
				}
			}
		}

		void Init(triangulateio& in) {
			in.pointlist = nullptr;
			in.pointattributelist = nullptr;
			in.pointmarkerlist = nullptr;
			in.numberofpoints = 0;
			in.numberofpointattributes = 0;

			in.trianglelist = nullptr;
			in.triangleattributelist = nullptr;
			in.trianglearealist = nullptr;
			in.neighborlist = nullptr;
			in.numberoftriangles = 0;
			in.numberofcorners = 0;
			in.numberoftriangleattributes = 0;

			in.segmentlist = nullptr;
			in.segmentmarkerlist = nullptr;
			in.numberofsegments = 0;

			in.holelist = nullptr;
			in.numberofholes = 0;

			in.regionlist = nullptr;
			in.numberofregions = 0;

			in.edgelist = nullptr;
			in.edgemarkerlist = nullptr;
			in.normlist = nullptr;
			in.numberofedges = 0;
		};

		Triangulator(triangulateio& in) { Init(in); }

		Triangulator(const XGLVertexList& v, ContourEndPoints& ce, triangulateio& in) {
			Init(in);
			in.numberofpoints = v.size();
			in.pointlist = (REAL*)malloc(sizeof(REAL) * 2 * in.numberofpoints);

			in.numberofsegments = v.size();
			in.segmentlist = (int*)malloc(sizeof(int) * 2 * in.numberofsegments);

			// build pointlist from XGLVertexList "v"
			unsigned int idx = 0;
			for (auto vrtx : v) {
				in.pointlist[idx * 2] = vrtx.v.x;
				in.pointlist[idx * 2 + 1] = vrtx.v.y;
				idx++;
			}
			
			//xyzzy
			int idx1 = 0;
			int idx2 = 1;
			idx = 0;
			for (auto vrtx : v) {
				in.segmentlist[idx * 2] = idx1;
				in.segmentlist[idx * 2 + 1] = idx2;
				idx++;
				idx1 = (idx1 + 1) % v.size();
				idx2 = (idx2 + 1) % v.size();
			}
		}
		void Dump(triangulateio& in) {
			xprintf("%d 2 %d 0\n", in.numberofpoints, in.numberofpointattributes);
			for (int i = 0; i < in.numberofpoints; i++)
				xprintf("%d %0.6f %0.6f\n", i+1, in.pointlist[i * 2], in.pointlist[i * 2 + 1]);

			xprintf("%d 0\n", in.numberofsegments);
			for (int i = 0; i < in.numberofsegments; i++)
				xprintf("%d %d %d\n", i+1, in.segmentlist[i * 2]+1, in.segmentlist[i * 2 + 1]+1);
		}
	};

	XGLFreeType(std::string text) : textToRender(text) {
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
			drawCurves = true;

			FT_Outline_Decompose(&g->outline, &fdc, this);
			contourEndPoints.push_back((int)v.size()-1);

			xprintf("There are %d contours:\n", contourEndPoints.size());
			for (auto c : contourEndPoints)
				xprintf("Contour@ %d\n", c);

			advance.x += g->advance.x;
			advance.y += g->advance.y;
		}

		numPoints = v.size();
		num2draw = numPoints;

		xprintf("found %d points\n", numPoints);

		// It's vital that these are initialized to empty!
		triangulateio in{ 0 }, out{ 0 };

		Triangulator t(v, contourEndPoints, in);

		xprintf("XGLFreeType::Triangulator()\n");
		t.Dump(in);

		triangulate("Vzp", &in, &out, nullptr);
	};

	~XGLFreeType() {
		FT_Done_Face(face);
		FT_Done_FreeType(ft);
	};

	void Draw() {
		if (v.size()) {
			int i = 0;
			for (auto c : contourEndPoints) {
				GLsizei count = c - i;
				count = num2draw;
				glDrawArrays(GL_LINE_LOOP, i, count);
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
		xprintf("MoveTo: %d %d\n", to->x, to->y);
		currentPoint = *to;

		if (v.size() == 0)
			firstPoint = *to;

		// if this isn't the very first vertex...
		if (v.size() > 0) {
			//...we've seen vertices, is this the very first contour?...
			if (contourEndPoints.empty())
				contourEndPoints.push_back((int)v.size());
			// not first contour ever, ensure it isn't the first contour of new glyph
			else if (v.size() > contourEndPoints.back())
				contourEndPoints.push_back((int)v.size());
		}

		// add the first point of the new contour
		v.push_back({ { Advance(to).x / scaleFactor, Advance(to).y / scaleFactor, 0 }, {}, {}, pointsColor });

		return 0;
	}

	int LineTo(const FT_Vector* to) {
		if (to->x != firstPoint.x || to->y != firstPoint.y) {
			xprintf("LineTo: %d %d\n", to->x, to->y);
			v.push_back({ { Advance(to).x / scaleFactor, Advance(to).y / scaleFactor, 0 }, {}, {}, pointsColor });
			currentPoint = *to;
		}
		else
			xprintf("LineTo: %d %d (coincident with firstPoint, ignoring)\n", to->x, to->y);

		return 0;
	}

	int ConicTo(const FT_Vector* control, const FT_Vector* to) {
		xprintf(" ConTo: %d %d\n", to->x, to->y);
		if (drawCurves)
			EvaluateQuadraticBezier(Advance(&currentPoint), Advance(control), Advance(to));
		else
			v.push_back({ { Advance(to).x / scaleFactor, Advance(to).y / scaleFactor, 0 }, {}, {}, pointsColor });
		currentPoint = *to;
		return 0;
	}

	int CubicTo(const FT_Vector* control1, const FT_Vector* control2, const FT_Vector* to) {
		xprintf("CubeTo: %d %d\n", to->x, to->y);
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

		// the triangulator() fails when there are coincident points, so avoid starting the curve at 0th point
		for (interpolant = interpolationFactor; interpolant < 1.0f; interpolant += interpolationFactor) {
			xa = GetInterpolatedPoint((float)p0.x, (float)p1.x, interpolant);
			ya = GetInterpolatedPoint((float)p0.y, (float)p1.y, interpolant);

			xb = GetInterpolatedPoint((float)p1.x, (float)p2.x, interpolant);
			yb = GetInterpolatedPoint((float)p1.y, (float)p2.y, interpolant);

			x = GetInterpolatedPoint(xa, xb, interpolant);
			y = GetInterpolatedPoint(ya, yb, interpolant);

			v.push_back({ { x / scaleFactor, y / scaleFactor, 0 }, {}, {}, pointsColor });
		}
	}

	void EvaluateCubicBezier(FT_Vector p0, FT_Vector p1, FT_Vector p2, FT_Vector p3) {
		float xa, xb, xc, ya, yb, yc;
		float xm, xn, ym, yn;
		float x, y;
		float interpolant;

		// the triangulator() fails when there are coincident points, so avoid starting the curve at 0th point
		for (interpolant = interpolationFactor; interpolant < 1.0f; interpolant += interpolationFactor) {
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
	}

	bool drawCurves;
	XGLColor pointsColor = XGLColors::yellow;
	XGLColor curvesColor = XGLColors::cyan;
	XGLColor controlColor = XGLColors::magenta;

	std::string textToRender;
	FreeTypeDecomposer fdc;
	ContourEndPoints contourEndPoints;
	FT_Library ft;
	FT_Face face;
	FT_GlyphSlot g;
	CharMap charMap;

	FT_Vector currentPoint;
	FT_Vector firstPoint;
	FT_Vector advance;

	float scaleFactor = 200.0f;
	float interpolationFactor = 0.5f;

	struct triangulateio in, mid, out;
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

	// now hook up the GUI sliders to the rotating torus thingy to control it's speeds.
	XGLGuiSlider *hs, *hs2;

	XGLGuiCanvas *sliders = (XGLGuiCanvas *)(GetGuiManager()->FindObject("HorizontalSliderWindow"));
	if (sliders != nullptr) {
		if ((hs = (XGLGuiSlider *)sliders->FindObject("Horizontal Slider 1")) != nullptr) {
			hs->AddMouseEventListener([hs](float x, float y, int flags) {
				if (hs->HasMouse()) {
					num2draw = (int)(hs->Position()*numPoints);
					xprintf("Position: %d\n", num2draw);
				}
			});
		}
		if ((hs2 = (XGLGuiSlider *)sliders->FindObject("Horizontal Slider 2")) != nullptr) {
			hs2->AddMouseEventListener([hs2](float x, float y, int flags) {
				if (hs2->HasMouse()) {
					num2draw2 = (int)(hs2->Position()*numPoints2);
					xprintf("Position: %d\n", num2draw2);
				}
			});
		}
	}
}
