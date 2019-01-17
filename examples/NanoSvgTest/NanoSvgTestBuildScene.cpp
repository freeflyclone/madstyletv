/**************************************************************
** NanoSvgTestBuildScene.cpp
**
** Just to demonstrate instantiation of a "ground"
** plane and a single triangle, with default camera manipulation
** via keyboard and mouse.
**************************************************************/
#include "ExampleXGL.h"

#define NANOSVG_ALL_COLOR_KEYWORDS	// Include full list of color keywords.
#define NANOSVG_IMPLEMENTATION		// Expands implementation
#include "nanosvg.h"
#include "nanosvgrast.h"

class NanoSVGShape : public XGLShape {
public:
	NanoSVGShape(std::string fileName) {
		image = nsvgParseFromFile(fileName.c_str(), "px", -4);
		for (NSVGshape* shape = image->shapes; shape != NULL; shape = shape->next) {
			for (NSVGpath* path = shape->paths; path != NULL; path = path->next) {
				float* p = path->pts;
				for (auto i = 0; i < path->npts; i++) {
					v.push_back({ { -p[0], p[1], 0 }, {}, {}, { 1, 1, 0, 1 } });
					p += 2;
				}
/*				for (auto i = 0; i < path->npts - 1; i += 3) {
					float* p = &path->pts[i * 2];
					v.push_back({ { -p[0], p[1], 0 }, {}, {}, { 1, 1, 0, 1 } });
					v.push_back({ { -p[2], p[3], 0 }, {}, {}, { 1, 1, 0, 1 } });
					v.push_back({ { -p[4], p[5], 0 }, {}, {}, { 1, 1, 0, 1 } });
					v.push_back({ { -p[6], p[7], 0 }, {}, {}, { 1, 1, 0, 1 } });
				}
*/
			}
		}
	}

	void Draw() {
		glDrawArrays(GL_LINE_STRIP, 0, GLsizei(v.size()));
		GL_CHECK("glDrawPoints() failed");
	}

	~NanoSVGShape() {
		nsvgDelete(image);
	}

private:
	struct NSVGimage* image;
};

extern bool initHmd;

void ExampleXGL::BuildScene() {
	XGLShape *shape;
	NanoSVGShape *svgShape;
	//initHmd = true;

	AddShape("shaders/000-simple", [&](){ shape = new XGLTriangle(); return shape; });
	AddShape("shaders/000-simple", [&](){ svgShape = new NanoSVGShape("../assets/letter.svg"); return svgShape; });
}
