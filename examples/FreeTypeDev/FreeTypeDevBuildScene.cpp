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
#include "XGLFreeType.h"

#include FT_OUTLINE_H


#ifdef FONT_NAME
#undef FONT_NAME
#endif

#define FONT_NAME "C:/windows/fonts/times.ttf"


int numPoints = 0;
int num2draw;

int numPoints2 = 0;
int num2draw2;

class XGLFreeType : public XGLShape {
public:
	typedef std::map<FT_ULong, FT_UInt> CharMap;
	typedef std::vector<GLsizei> ContourEndPoints;

	// use the "Triangle" package from http://www.cs.cmu.edu/~quake/triangle.html
	class Triangulator {
	public:
		Triangulator(const XGLVertexList& v, ContourEndPoints& ce, triangulateio& in) {
			in.numberofpoints = (int)v.size();
			in.pointlist = (REAL*)malloc(sizeof(REAL) * 2 * in.numberofpoints);

			in.numberofsegments = (int)v.size();
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
		FT_Set_Char_Size(face, ftSize, ftSize, ftResolution, ftResolution);

		FT_GlyphSlot g = face->glyph;

		// build an XGLCharMap of the entire set of glyphs for this font.
		// (this could be huge for Chinese fonts)
		for (charcode = FT_Get_First_Char(face, &gindex); gindex; charcode = FT_Get_Next_Char(face, charcode, &gindex))
			charMap.emplace(charcode, gindex);

		const int numGlyphs = (const int)(charMap.size());

		currentPoint = { 0, 0 };
		advance = { 0, 0 };

		drawCurves = true;

		for (auto c : textToRender) {
			gindex = charMap[c];

			FT_Load_Glyph(face, gindex, FT_LOAD_FORCE_AUTOHINT | FT_LOAD_TARGET_NORMAL);
			FT_Outline_Decompose(&g->outline, &fdc, &fdc);

			FT::GlyphOutline glyph = fdc.Outline();
			int i = 0;
			for (auto c : glyph) {
				xprintf("Countour %d has %d points\n", i, c.size());
				for (auto vec : c) {
					v.push_back({ { vec.x / scaleFactor, vec.y / scaleFactor, 0 }, {}, {}, { XGLColors::yellow } });
					xprintf(" %0.4f,%0.4f\n", vec.x/scaleFactor, vec.y/scaleFactor);
				}
				break;
				i++;
			}

			advance.x += g->advance.x;
			advance.y += g->advance.y;
		}

		numPoints = (int)v.size();
		num2draw = numPoints;
	};

	~XGLFreeType() {
		FT_Done_Face(face);
		FT_Done_FreeType(ft);
	};

	void Draw() {
		if (v.size()) {
			glDrawArrays(GL_LINE_STRIP, 0, num2draw);
			GL_CHECK("glDrawArrays() failed");
		}
	}

	FT_Vector Advance(const FT_Vector* vector) {
		return{ advance.x + vector->x, advance.y + vector->y };
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

			tIn.push_back({ { x / scaleFactor, y / scaleFactor, 0 }, {}, {}, pointsColor });
		}
		if (p2.x != firstPoint.x || p2.y != firstPoint.y)
			tIn.push_back({ { p2.x / scaleFactor, p2.y / scaleFactor, 0 }, {}, {}, pointsColor });
		else
			xprintf("Final point of Quadratic Bezier is coincident with contour start, ignoring\n");
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

			tIn.push_back({ { x / scaleFactor, y / scaleFactor, 0 }, {}, {}, curvesColor });
		}
		if (p3.x != firstPoint.x || p3.y != firstPoint.y)
			tIn.push_back({ { p3.x / scaleFactor, p3.y / scaleFactor, 0 }, {}, {}, pointsColor });
		else
			xprintf("Final point of Cubic Bezier is coincident with contour start, ignoring\n");
	}

	void RenderTriangles(triangulateio& in) {
		for (int i = 0; i < in.numberoftriangles; i++) {
			for (int j = 0; j < 3; j++) {
				int idx = in.trianglelist[i * 3 + j];
				// modulo trick: get the next (possibly wrapped) vertex of *this* triangle
				int idxNext = in.trianglelist[(i * 3 + ((j + 1) % 3))];

				// first point of the line segment of this triangle's edge
				REAL x = in.pointlist[idx * 2];
				REAL y = in.pointlist[idx * 2 + 1];
				v.push_back({ { x, y, 0 }, {}, {}, { XGLColors::yellow } });

				// for debugging during dev
				if (drawMode == GL_LINES) {
					REAL x2 = in.pointlist[idxNext * 2];
					REAL y2 = in.pointlist[idxNext * 2 + 1];
					v.push_back({ { x2, y2, 0 }, {}, {}, { XGLColors::yellow } });
				}
			}
		}
	}

	const FT_F26Dot6 ftSize{ 256 };
	const FT_UInt ftResolution{ 512 };

	bool drawCurves;
	GLuint drawMode = GL_TRIANGLES; // GL_LINES or GL_TRIANGES (for filling in)

	XGLColor pointsColor = XGLColors::yellow;
	XGLColor curvesColor = XGLColors::cyan;
	XGLColor controlColor = XGLColors::magenta;
	XGLVertexList tIn;  //trianulator() input
	XGLVertexList contoursEnds;

	std::string textToRender;
	FT::GlyphDecomposer fdc;
	ContourEndPoints contourEndPoints;
	FT_Library ft;
	FT_Face face;
	FT_GlyphSlot g;
	CharMap charMap;

	FT_Vector currentPoint;
	FT_Vector firstPoint;
	FT_Vector advance;

	float scaleFactor = 200.0f;
	float interpolationFactor = 0.2f;

	struct triangulateio in, mid, out;
};

void ExampleXGL::BuildScene() {
	XGLFreeType *shape;
	//Triangulator *t;

	// Initialize the Camera matrix
	glm::vec3 cameraPosition(0, -0.1, 14);
	glm::vec3 cameraDirection = glm::normalize(cameraPosition*-1.0f);
	glm::vec3 cameraUp = { 0, 0, 1 };
	camera.Set(cameraPosition, cameraDirection, cameraUp);

	AddShape("shaders/000-simple", [&](){ shape = new XGLFreeType(config.WideToBytes(config.Find(L"FreeTypeText")->AsString())); return shape; });
	glm::mat4 translate = glm::translate(glm::mat4(), glm::vec3(-2, -2, 1.0));
	shape->model = translate;
	
	// now hook up the GUI sliders to the rotating torus thingy to control it's speeds.
	XGLGuiSlider *hs, *hs2;

	XGLGuiCanvas *sliders = (XGLGuiCanvas *)(GetGuiManager()->FindObject("HorizontalSliderWindow"));
	if (sliders != nullptr) {
		if ((hs = (XGLGuiSlider *)sliders->FindObject("Horizontal Slider 1")) != nullptr) {
			hs->AddMouseEventListener([hs](float x, float y, int flags) {
				if (hs->HasMouse()) {
					num2draw = (int)(hs->Position()*numPoints);
				}
			});
		}
		if ((hs2 = (XGLGuiSlider *)sliders->FindObject("Horizontal Slider 2")) != nullptr) {
			hs2->AddMouseEventListener([hs2](float x, float y, int flags) {
				if (hs2->HasMouse()) {
					num2draw2 = (int)(hs2->Position()*numPoints2);
				}
			});
		}
	}
}
