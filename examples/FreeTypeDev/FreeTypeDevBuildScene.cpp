/**************************************************************
** FreeTypeDevBuildScene.cpp
**
** Demonstrate drawing FreeType outlines using 
** FT_Outline_Decompose. Found the EvaluateXXXBezier() methods
** on StackOverFlow.
**************************************************************/
#include "ExampleXGL.h"
#include <string>

#define REAL double
#define TRILIBRARY
#define ANSI_DECLARATORS

extern "C" {
#include "triangle.h"
}

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
	class TriangulatorConverter : public triangulateio {
	public:
		void Init() {
			pointlist = 0;
			pointattributelist = 0;
			pointmarkerlist = 0;
			numberofpoints = 0;
			numberofpointattributes = 0;
			trianglelist = 0;
			triangleattributelist = 0;
			trianglearealist = 0; 
			neighborlist = 0;     
			numberoftriangles = 0;
			numberofcorners = 0;
			numberoftriangleattributes = 0;
			segmentlist = 0;
			segmentmarkerlist = 0;
			numberofsegments = 0;
			holelist = 0;
			numberofholes = 0;
			regionlist = 0;
			numberofregions = 0;
			edgelist = 0;
			edgemarkerlist = 0;
			normlist = 0;
			numberofedges = 0;

		}
		TriangulatorConverter(FT::GlyphOutline& go, REAL scaleFactor) {
			Init();

			int numPoints = 0, numSegments;

			for (auto c : go)
				numPoints += c.size();

			numSegments = numPoints;

			pointlist = (REAL*)malloc(2 * sizeof(REAL) * numPoints);
			segmentlist = (int*)malloc(2 * sizeof(int) * numSegments);
			holelist = (REAL*)malloc(2 * sizeof(REAL) * 20);

			int contourOffset = 0;
			//FT::Contour c = go[0];
			for (auto c : go)
			{
				int numPoints = c.size();
				int numSegments = numPoints;

				for (int i = 0; i < numPoints; i++) {
					pointlist[i * 2 + contourOffset] = c[i].v.x / scaleFactor;
					pointlist[i * 2 + 1 + contourOffset] = c[i].v.y / scaleFactor;
				}

				for (int i = 0; i < numPoints; i++) {
					int j = (i + 1) % numPoints;
					segmentlist[i * 2 + contourOffset] = i;
					segmentlist[i * 2 + 1 + contourOffset] = j;
				}
				numberofpoints += numPoints;
				numberofsegments += numSegments;

				contourOffset += numPoints;
			}
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
		advance = { 0, 0 };

		for (auto c : textToRender) {
			gindex = charMap[c];

			// load the glyph from the font
			FT_Load_Glyph(face, gindex, FT_LOAD_FORCE_AUTOHINT | FT_LOAD_TARGET_NORMAL);

			// decompose the outline using the spiffy new GlyphDecomposer class
			FT_Outline_Decompose(&g->outline, &fdc, &fdc);

			// get the GlyphDecomposer's GlyphOutline (consisting of 1 or more Contours)
			FT::GlyphOutline glyphOutline = fdc.Outline();

			TriangulatorConverter t(glyphOutline, scaleFactor);

			out = {};

			triangulate("qzp", (triangulateio*)&t, &out, NULL);

			RenderTriangles(out);

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
			glDrawArrays(GL_TRIANGLES, 0, num2draw);
			GL_CHECK("glDrawArrays() failed");
		}
	}

	FT_Vector Advance(const FT_Vector& vector) {
		return{ advance.x + vector.x, advance.y + vector.y };
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

	const FT_F26Dot6 ftSize{ 1026 };
	const FT_UInt ftResolution{ 2048 };

	GLuint drawMode = GL_TRIANGLES; // GL_LINES or GL_TRIANGES (for filling in)
	XGLVertexList tIn;  //trianulator() input

	std::string textToRender;
	FT::GlyphDecomposer fdc;
	FT_Library ft;
	FT_Face face;
	FT_GlyphSlot g;
	CharMap charMap;

	FT_Vector advance;

	float scaleFactor = 1600.0f;

	struct triangulateio in, out;
};

void ExampleXGL::BuildScene() {
	XGLFreeType *shape;

	// Initialize the Camera matrix
	glm::vec3 cameraPosition(0, -5, 14);
	glm::vec3 cameraDirection = glm::normalize(cameraPosition*-1.0f);
	glm::vec3 cameraUp = { 0, 0, 1 };
	camera.Set(cameraPosition, cameraDirection, cameraUp);

	AddShape("shaders/000-simple", [&](){ shape = new XGLFreeType(config.WideToBytes(config.Find(L"FreeTypeText")->AsString())); return shape; });
	glm::mat4 translate = glm::translate(glm::mat4(), glm::vec3(-2, -2, 1.0));
	shape->model = translate;
	
	// now hook up the GUI sliders to the rotating torus thingy to control it's speeds.
	XGLGuiSlider *hs;

	XGLGuiCanvas *sliders = (XGLGuiCanvas *)(GetGuiManager()->FindObject("HorizontalSliderWindow"));
	if (sliders != nullptr) {
		if ((hs = (XGLGuiSlider *)sliders->FindObject("Horizontal Slider 1")) != nullptr) {
			hs->AddMouseEventListener([hs](float x, float y, int flags) {
				if (hs->HasMouse()) {
					num2draw = (int)(hs->Position()*numPoints);
				}
			});
		}
	}
}
