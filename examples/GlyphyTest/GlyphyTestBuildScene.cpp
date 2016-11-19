/**************************************************************
** GlyphyTestBuildScene.cpp
**
** Just to demonstrate instantiation of a "ground"
** plane and a single triangle, with default camera manipulation
** via keyboard and mouse.
**************************************************************/
#include "ExampleXGL.h"

#include "config.h"
#include "demo-buffer.h"
#include "demo-font.h"
#include "demo-view.h"

static demo_glstate_t *st;
static demo_view_t *vu;
static demo_buffer_t *buffer;

void ExampleXGL::BuildScene() {
	XGLShape *shape;

	st = demo_glstate_create();
	//vu = demo_view_create(st);
	//demo_view_print_help(vu);

	AddShape("shaders/simple", [&](){ shape = new XGLTriangle(); return shape; });
}
