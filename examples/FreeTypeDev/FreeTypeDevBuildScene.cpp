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
	class TriangulatorConverter {
	public:
		void Init(triangulateio& in) {
			in.pointlist = 0;
			in.pointattributelist = 0;
			in.pointmarkerlist = 0;
			in.numberofpoints = 0;
			in.numberofpointattributes = 0;
			in.trianglelist = 0;
			in.triangleattributelist = 0;
			in.trianglearealist = 0;
			in.neighborlist = 0;
			in.numberoftriangles = 0;
			in.numberofcorners = 0;
			in.numberoftriangleattributes = 0;
			in.segmentlist = 0;
			in.segmentmarkerlist = 0;
			in.numberofsegments = 0;
			in.holelist = 0;
			in.numberofholes = 0;
			in.regionlist = 0;
			in.numberofregions = 0;
			in.edgelist = 0;
			in.edgemarkerlist = 0;
			in.normlist = 0;
			in.numberofedges = 0;

		}
		TriangulatorConverter(FT::GlyphOutline& go, triangulateio&  in, XGLVertex& a, REAL scaleFactor) {
			Init(in);

			size_t numPoints = 0, numSegments;

			for (auto c : go)
				numPoints += c.v.size();

			numSegments = numPoints;

			in.pointlist = (REAL*)malloc(2 * sizeof(REAL) * numPoints);
			in.segmentlist = (int*)malloc(2 * sizeof(int) * numSegments);
			in.holelist = (REAL*)malloc(2 * sizeof(REAL) * 20);

			int contourOffset = 0;
			int pIdx = 0;
			int sIdx = 0;
			for (auto c : go)
			{
				size_t numPoints = c.v.size();
				size_t numSegments = numPoints;

				for (int i = 0; i < numPoints; i++) {
					in.pointlist[pIdx++] = c.v[i].v.x / scaleFactor + a.x;
					in.pointlist[pIdx++] = c.v[i].v.y / scaleFactor + a.y;
				}

				for (int i = 0; i < numPoints; i++) {
					int j = (i + 1) % numPoints;
					in.segmentlist[sIdx++] = i + contourOffset;
					in.segmentlist[sIdx++] = j + contourOffset;
				}
				in.numberofpoints += (int)numPoints;
				in.numberofsegments += (int)numSegments;

				bool isClockwise;
				XGLVertex v = c.ComputeCentroid(&isClockwise);
				if (!isClockwise) {
					in.holelist[in.numberofholes * 2] = v.x / scaleFactor + a.x;
					in.holelist[in.numberofholes * 2 + 1] = v.y / scaleFactor + a.y;
					in.numberofholes++;
				}
				contourOffset += (int)numPoints;
			}
		}
		void Dump(triangulateio& in) {
			xprintf("numberofpoints: %d\n", in.numberofpoints);
			for (int i = 0; i < in.numberofpoints; i++)
				xprintf("  %0.5f, %0.5f\n", in.pointlist[i * 2], in.pointlist[i * 2 + 1]);

			xprintf("numberofsegments: %d\n", in.numberofsegments);
			for (int i = 0; i < in.numberofsegments; i++)
				xprintf("  %d, %d\n", in.segmentlist[i * 2], in.segmentlist[i * 2 + 1]);

			xprintf("numberofholes: %d\n", in.numberofholes);
			for (int i = 0; i < in.numberofholes; i++)
				xprintf("  %0.5f, %0.5f\n", in.holelist[i * 2], in.segmentlist[i * 2 + 1]);
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
		advance = { 0, 0, 0 };

		for (auto c : textToRender) {
			if (c != ' ') {
				gindex = charMap[c];

				// load the glyph from the font
				FT_Load_Glyph(face, gindex, FT_LOAD_FORCE_AUTOHINT | FT_LOAD_TARGET_NORMAL);

				// decompose the outline using the spiffy new GlyphDecomposer class
				fdc.Reset();
				FT_Outline_Decompose(&g->outline, &fdc, &fdc);

				// get the GlyphDecomposer's GlyphOutline (consisting of 1 or more Contours)
				FT::GlyphOutline& glyphOutline = fdc.Outline();

				in = {};
				out = {};
				TriangulatorConverter t(glyphOutline, in, advance, scaleFactor);
				t.Init(out);

				t.Dump(in);
				//triangulate("q25zp", &in, &out, NULL);
				triangulate("zpYY", &in, &out, NULL);

				RenderTriangles(out);
			}
			advance.x += g->advance.x / scaleFactor;
			advance.y += g->advance.y / scaleFactor;
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

	XGLVertex Advance(const XGLVertex& vector) {
		return { advance.x + vector.x, advance.y + vector.y, 0 };
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

	const FT_F26Dot6 ftSize{ 1024 };
	const FT_UInt ftResolution{ 1024 };

	GLuint drawMode = GL_TRIANGLES; // GL_LINES or GL_TRIANGES (for filling in)
	XGLVertexList tIn;  //trianulator() input

	std::string textToRender;
	FT::GlyphDecomposer fdc;
	FT_Library ft;
	FT_Face face;
	FT_GlyphSlot g;
	CharMap charMap;

	XGLVertex advance;

	float scaleFactor = 3276.80f;

	struct triangulateio in, out;
};

void ExampleXGL::BuildScene() {
	XGLFreeType *shape;

	// Initialize the Camera matrix
	glm::vec3 cameraPosition(0, -5, 8);
	glm::vec3 cameraDirection = glm::normalize(cameraPosition*-1.0f);
	glm::vec3 cameraUp = { 0, 0, 1 };
	camera.Set(cameraPosition, cameraDirection, cameraUp);

	AddShape("shaders/bezierPix", [&](){ shape = new XGLFreeType(config.WideToBytes(config.Find(L"FreeTypeText")->AsString())); return shape; });
	glm::mat4 translate = glm::translate(glm::mat4(), glm::vec3(-2, 0, 0));
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
