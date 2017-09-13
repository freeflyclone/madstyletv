/**************************************************************
** WorldGUIDevBuildScene.cpp
**
** Just to demonstrate instantiation of a "ground" plane and an
** XGLGuiCanvas for VR GUI development.  Heavily commented as
** a product of reverse-engineering my own work ;)
**************************************************************/
#include "ExampleXGL.h"

void ExampleXGL::BuildScene() {
	XGLGuiCanvas *shape;

	// XGLGuiCanvas is intended to be a textured quadrilateral that serves as an overlay for
	// 2D GUI elements to be rendered on.  Think of it as the equivalent of a computer monitor,
	// but in world space.
	//
	// The width & height arguments are the dimensions of the buffer that is used as a GL texture
	// for 2D grapics operations, and they are ALSO the dimensions of the geometry of the quadrilateral.
	// This leads to the unfortunate side-effect of being HUGE in world space by default.  This can be
	// fixed by applying a scale factor to the model matrix.  Could do that in the constructor I suppose.
	//
	// The choice of 1920 x 1080 gives texture buffer dimensions equivalent to real world HDTV dimensions,
	// With OVR, 1 unit in world space equals 1 meter in physical space, so that means this XGLGuiCanvas
	// is nearly 2 kilometers wide, before scaling.  Scaling by 0.01 reduces it to 19.2 x 10.8 meters.
	// Nice and big but not gianormous.
	AddShape("shaders/gui-tex", [&shape,this](){
		shape = new XGLGuiCanvas(this, 1920,1080); 
		return shape; 
	});
	glm::mat4 scale;
	glm::mat4 rotate;
	glm::mat4 translate;

	// scale it to "reasonable" world dimensions.
	scale = glm::scale(glm::mat4(), glm::vec3(0.01, 0.01, 0.01));

	// flip it up so it's vertical
	rotate = glm::rotate(glm::mat4(), glm::radians(90.0f), glm::vec3(1.0f, 0.0f, 0.0f));

	// move it so the bottom edge is on the ground plane, and it's centered horizontally.
	translate = glm::translate(glm::mat4(), glm::vec3(-9.6, 0.0, 10.8));

	// applay all those to the modle matrix.
	shape->model = translate * rotate * scale;

	// XGLGuiCanvas supports "RenderText", using XGLFont to actually render text.  It does bit-blitting
	// CPU side of individual glyphs from the FreeType representation.  It's not very sophisticated, possible
	// problems include boundary checking issues.
	// RenderText() has a default pixel size of 64, which makes for fairly large text.  It's a somewhat arbitrary
	// choice, balancing size vs "fuzziness" caused by anti-aliasing by FreeType blitting.
	//
	// XGLGuiCanvas also supports a "pen", or cursor, which gets updated by RenderText(), so that consecutive
	// RenderText() calls provide expected line-oriented behavior.  It is a 2D coordinate, in pixels. 
	// For RenderText() it is the upper-left corner of the box of text that is rendered in each call.
	// The dimensions of that box are set by the "pixelSize" argument, and the number of characters to be
	// rendered.
	//
	// Linefeeds do what one expects.  The amount of vertical offset caused by a linefeed is dependent 
	// on the "pixelSize" argument to the RenderText() call.
	shape->RenderText("This is a test.\n");
	shape->RenderText("Another line, 64 pixels (the default) tall");
	shape->RenderText("\nThis is a smaller test", 16);

	// The SetPenPosition() call allows one to set the current pen position.
	// Linefeeds reset it to the left-margin, but still provide the same vertical offset behavior.
	shape->SetPenPosition(960, 540);
	shape->RenderText("Text in the middle\n", 32);
	shape->RenderText("Another line after a line-feed", 32);

	// "shaders/gui-tex" does appropriate world-space projections, just like "shaders/specular".  The fragment
	// shader by default expects that the texture buffer is 8-bit gray-scale (for anti-aliasing).  The fragment
	// shader uses attributes.diffuseColor as the "foreground" color, and attributes.ambientColor as the "background"
	// color.  The alpha component allows for transparency, like one would expect.
	shape->attributes.ambientColor = { 1.0, 0.0, 1.0, 0.1 };
	shape->attributes.diffuseColor = { 1.0, 1.0, 1.0, 1.0 };
}
