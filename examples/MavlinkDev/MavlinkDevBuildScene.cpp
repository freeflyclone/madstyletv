/**************************************************************
** MavlinkDevBuildScene.cpp
**
** Demonstrate instantiation of XMavlink object in the
** XGL framework
**************************************************************/
#include "ExampleXGL.h"
#include "xmavlink.h"

XMavlink *mavlink;

void ExampleXGL::BuildScene() {
	XGLShape *shape;

	AddShape("shaders/000-simple", [&](){ shape = new XGLTriangle(); return shape; });

	try {
		mavlink = new XMavlink("\\\\.\\COM17");
	}
	catch (std::runtime_error e) {
		xprintf("Well that didn't work out: %s\n", e.what());
	}
}
