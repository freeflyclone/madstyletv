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

void ExampleXGL::BuildScene() {
	XGLShape *shape;

	AddShape("shaders/000-simple", [&](){ shape = new XGLTriangle(); return shape; });

	struct NSVGimage* image;
	image = nsvgParseFromFile("../assets/e-man-tv.svg", "px", 96);
	xprintf("size: %f x %f\n", image->width, image->height);
	// Use...
	for (auto shape = image->shapes; shape != NULL; shape = shape->next) {
		xprintf("shape\n");
		for (auto path = shape->paths; path != NULL; path = path->next) {
			xprintf("path\n");
			for (auto i = 0; i < path->npts - 1; i += 3) {
				float* p = &path->pts[i * 2];
				xprintf("%d points\n", path->npts);
				//drawCubicBez(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7]);
			}
		}
	}
	// Delete
	nsvgDelete(image);
}
