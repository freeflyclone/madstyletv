#include "ExampleXGL.h"

XGLGuiManager *appGuiManager;

void ExampleXGL::BuildGUI() {
	glm::mat4 scale, rotate, translate;
	XGLGuiCanvas *gc;

	// Create a GUI manager, set it to not visible initially.
	// Normally XGLGuiManager shapes are special and are added with AddGuiShape(), but for this app
	// it's part of the virtual space and so we want to treat it as a normal shape.
	CreateShape("shaders/ortho", [&]() { appGuiManager = new XGLGuiManager(this); return appGuiManager; }, 2);
	appGuiManager->isVisible = false;

	scale = glm::scale(glm::mat4(), glm::vec3(0.001, 0.001, 0.001));
	rotate = glm::rotate(glm::mat4(), glm::radians(90.0f), glm::vec3(1.0f, 0.0f, 0.0f));
	translate = glm::translate(glm::mat4(), glm::vec3(-.96, 1.0, 1.08*0.5));
	appGuiManager->model = translate * rotate * scale;


	// XGLGuiCanvas is intended to be a textured quadrilateral that serves as an overlay for
	// 2D GUI elements to be rendered on.  Think of it as the equivalent of a computer monitor,
	// but in world space.
	//
	// The width & height arguments are the dimensions of the buffer that is used as a GL texture
	// for 2D grapics operations, and they are ALSO the dimensions of the geometry of the quadrilateral.
	// This leads to the unfortunate side-effect of being HUGE in world space by default.  This can be
	// fixed by applying a scale factor to the model matrix of the parent XGLGuiManager shape.
	/// Could do that in the constructor I suppose.
	//
	// The choice of 1920 x 1080 gives texture buffer dimensions equivalent to real world HDTV dimensions,
	// With OVR, 1 unit in world space equals 1 meter in physical space, so that means this XGLGuiCanvas
	// is nearly 2 kilometers wide, before scaling.
	// Scaling by 0.001 reduces it to 1.92 x 1.08 meters. Nice and big but not gianormous.
	CreateShape("shaders/gui-tex", [&, this](){	gc = new XGLGuiCanvas(this, 1920, 1080); return gc;	});

	// disable depth buffer writing for the GUI canvas.  Thus it won't occlude things drawn after,
	// like the objects representing the Touch controllers.
	gc->preRenderFunction = [](float clock) {
		glDepthMask(GL_FALSE);
		GL_CHECK("glDepthMask(GL_FALSE) failed");
	};
	gc->postRenderFunction = [](float clock) {
		glDepthMask(GL_TRUE);
		GL_CHECK("glDepthMask(GL_TRUE) failed");
	};

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
	//
	// RenderText() checks to see if the next character will cross the right window boudary, and will
	// truncate the text from that letter on.
	gc->RenderText("This is a test.\n");
	gc->RenderText("Another line, 64 pixels (the default) tall, that's got a bunch of text to see how I handle hitting the right margin.");
	gc->RenderText("\nThis is a smaller test", 16);

	// The SetPenPosition() call allows one to set the current pen position.
	// Linefeeds reset it to the left-margin, but still provide the same vertical offset behavior.
	gc->SetPenPosition(960, 540);
	gc->RenderText("Text in the middle\n", 32);
	gc->RenderText("Another line after a line-feed", 32);

	// "shaders/gui-tex" does appropriate world-space projections, just like "shaders/specular".  The fragment
	// shader by default expects that the texture buffer is 8-bit gray-scale (for anti-aliasing).  The fragment
	// shader uses attributes.diffuseColor as the "foreground" color, and attributes.ambientColor as the "background"
	// color.  The alpha component allows for transparency,	as one would expect.
	gc->attributes.ambientColor = { 0.001, 0.001, 0.001, 0.5 };
	gc->attributes.diffuseColor = { 1.0, 1.0, 1.0, 1.0 };

	appGuiManager->AddChild(gc);

	return;
}
