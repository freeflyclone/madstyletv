/**************************************************************
** OVRTestBuildScene.cpp
**
** Build a scene with some stuff, and introduce a "sled" for
** the HMD and touch controllers.
**************************************************************/
#include "ExampleXGL.h"

// these needs to be file scope at least.  Local (to ::BuildScene) doesn't work
// Reason: if you "capture by reference" in a lambda function, they need to
// be valid.  If they're local variables, "capture by reference" doesn't work
// if the lambda is being called outside the scope of the function that created
// it.  Which is almost always the case the way lambda's get used herein.
XGLSphere *sphere;
XGLShape  *hmdSled;
XGLGuiCanvas *guiCanvas;
XGLGuiManager *gm;
XGLGuiCanvas *gc;


const float constSpeed1 = 60.0f * 4.0f;
const float constSpeed2 = 45.0f * 4.0f;
const float constSpeed3 = 30.0f * 4.0f;

float speed1 = constSpeed1;
float speed2 = constSpeed2;
float speed3 = constSpeed3;

void ExampleXGL::BuildScene() {
	XGLShape *shape, *child1, *child2, *child3, *child4;
	XGLShape *hmdChild;
	glm::mat4 rotate, translate;

	AddShape("shaders/specular", [&](){ shape = new XGLTorus(5.0f, 1.0f, 64, 32); return shape; });
	shape->attributes.diffuseColor = { 0.005, 0.005, 0.005, 1 };
	shape->SetAnimationFunction([shape](float clock) {
		glm::mat4 rotate = glm::rotate(glm::mat4(), clock / speed1, glm::vec3(1.0f, 0.0f, 0.0f));
		shape->model = rotate;
	});
	CreateShape("shaders/specular", [&](){ child3 = new XGLTransformer(); return child3; });
	child3->SetAnimationFunction([child3](float clock) {
		float translateFunction = sin(clock / 180.0f);
		glm::mat4 rotate = glm::rotate(glm::mat4(), clock / speed2, glm::vec3(0.0f, 0.0f, 1.0f));
		child3->model = rotate;
	});
	shape->AddChild(child3);

	CreateShape("shaders/specular", [&](){ child4 = new XGLTorus(2.0f, 0.5f, 64, 32); return child4; });
	child4->attributes.diffuseColor = { 1.0, 0.00001, 0.00001, 1 };
	translate = glm::translate(glm::mat4(), glm::vec3(5.0, 0, 0));
	rotate = glm::rotate(glm::mat4(), glm::radians(90.0f), glm::vec3(1.0f, 0.0f, 0.0f));
	child4->model = translate * rotate;

	CreateShape("shaders/000-simple", [&](){ child1 = new XGLTransformer(); return child1; });
	child1->SetAnimationFunction([child1](float clock) {
		float translateFunction = sin(clock / 180.0f);
		glm::mat4 rotate = glm::rotate(glm::mat4(), clock / speed3, glm::vec3(0.0f, 0.0f, 1.0f));
		child1->model = rotate;
	});

	CreateShape("shaders/specular", [&](){ child2 = new XGLTorus(0.75f, 0.25f, 64, 32); return child2; });
	child2->attributes.diffuseColor = (XGLColors::yellow);
	translate = glm::translate(glm::mat4(), glm::vec3(2.0, 0, 0));
	rotate = glm::rotate(glm::mat4(), glm::radians(90.0f), glm::vec3(1.0f, 0.0f, 0.0f));
	child2->model = translate * rotate;

	child1->AddChild(child2);
	child4->AddChild(child1);
	child3->AddChild(child4);

	AddShape("shaders/specular", [&](){ shape = new XGLTorus(3.0f, 0.5f, 64, 32); return shape; });
	shape->attributes.diffuseColor = (XGLColors::blue);
	shape->model = glm::translate(glm::mat4(), glm::vec3(20, 0, 0));

	AddShape("shaders/specular", [&](){ shape = new XGLTorus(3.0f, 0.5f, 64, 32); return shape; });
	shape->attributes.diffuseColor = (XGLColors::red);
	shape->model = glm::translate(glm::mat4(), glm::vec3(-20, 0, 0))
		* glm::scale(glm::mat4(), glm::vec3(2, 2, 2));

	AddShape("shaders/specular", [&](){ shape = new XGLTorus(3.0f, 0.5f, 64, 32); return shape; });
	shape->attributes.diffuseColor = (XGLColors::green);
	shape->model = glm::translate(glm::mat4(), glm::vec3(30, 0, 0))
		* glm::rotate(glm::mat4(), glm::radians(90.0f), glm::vec3(0, 1, 0))
		* glm::scale(glm::mat4(), glm::vec3(2, 2, 2));

	XInputKeyFunc renderMod = [&](int key, int flags) {
		const bool isDown = (flags & 0x8000) == 0;
		const bool isRepeat = (flags & 0x4000) != 0;
		static bool wireFrameMode = false;

		if (isDown && !isRepeat){
			wireFrameMode = wireFrameMode ? false : true;
			glPolygonMode(GL_FRONT_AND_BACK, wireFrameMode ? GL_LINE : GL_FILL);
			GL_CHECK("glPolygonMode() failed.");
		}
	};

	AddKeyFunc('M', renderMod);
	AddKeyFunc('m', renderMod);

	// HMD "sled" provides a single anchor point in world space for the HMD and Touch controllers
	// and anything else that needs to rendered as part of the user's "personal space".
	//
	// Could be a car interior, an aircraft/spacecraft cockpit, or what have you.
	//
	// NOTE: the way it works now, only "top-level" objects (created by AddShape()) will be
	//       affected by the optional "layer" argument.  All child objects thereof will
	//       rendered at that object's layer's time.
	AddShape("shaders/000-simple", [&]() {hmdSled = new XGLTransformer(); return hmdSled; }, 2);
	hmdSled->SetName("HmdSled");

	CreateShape("shaders/specular", [&](){ hmdChild = new XGLSphere(0.1f, 32); hmdChild->attributes.diffuseColor = XGLColors::cyan;  return hmdChild; }, 2);
	//hmdChild->attributes.diffuseColor = XGLColors::cyan;
	hmdSled->AddChild(hmdChild);

	CreateShape("shaders/specular", [&]() { shape = new XGLSphere(0.05f, 16); return shape; });
	shape->SetName("LeftHand");
	hmdSled->AddChild(shape);

	CreateShape("shaders/specular", [&]() { shape = new XGLSphere(0.05f, 16); return shape; });
	shape->SetName("RightHand");
	hmdSled->AddChild(shape);

	// the XGLGuiManager() serves as the root of the GuiShape tree, and intercepts the '~' key
	// for activation of the GUI (a la Id games).
	CreateShape("shaders/ortho", [&]() { gm = new XGLGuiManager(this); return gm; });

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
	CreateShape("shaders/gui-tex", [this](){
		gc = new XGLGuiCanvas(this, 1920, 1080);
		return gc;
	}, 2);
	gc->preRenderFunction = [](float clock) {
		glDepthMask(GL_FALSE);
		GL_CHECK("glDepthMask(GL_FALSE) failed");
	};
	gc->postRenderFunction = [](float clock) {
		glDepthMask(GL_TRUE);
		GL_CHECK("glDepthMask(GL_TRUE) failed");
	};

	gm->AddChild(gc);
	glm::mat4 scale;

	// scale it to "reasonable" world dimensions.
	scale = glm::scale(glm::mat4(), glm::vec3(0.001, 0.001, 0.001));

	// flip it up so it's vertical
	rotate = glm::rotate(glm::mat4(), glm::radians(90.0f), glm::vec3(1.0f, 0.0f, 0.0f));

	// move it so the bottom edge is on the ground plane, and it's centered horizontally.
	translate = glm::translate(glm::mat4(), glm::vec3(-.96, 1.0, 1.08*1.5));

	// applay all those to the modle matrix.
	gc->model = translate * rotate * scale;

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
	gc->RenderText("This is a test.\n");
	gc->RenderText("Another line, 64 pixels (the default) tall");
	gc->RenderText("\nThis is a smaller test", 16);

	// The SetPenPosition() call allows one to set the current pen position.
	// Linefeeds reset it to the left-margin, but still provide the same vertical offset behavior.
	gc->SetPenPosition(960, 540);
	gc->RenderText("Text in the middle\n", 32);
	gc->RenderText("Another line after a line-feed", 32);

	// "shaders/gui-tex" does appropriate world-space projections, just like "shaders/specular".  The fragment
	// shader by default expects that the texture buffer is 8-bit gray-scale (for anti-aliasing).  The fragment
	// shader uses attributes.diffuseColor as the "foreground" color, and attributes.ambientColor as the "background"
	// color.  The alpha component allows for transparency, like one would expect.
	gc->attributes.ambientColor = { 1.0, 0.0, 1.0, 0.1 };
	gc->attributes.diffuseColor = { 1.0, 1.0, 1.0, 1.0 };

	// attach the guiCanvas to the sled
/*	if ((guiCanvas = (XGLGuiCanvas *)FindObject("XGLGuiCanvas1")) != nullptr) {
		guiCanvas->SetAnimationFunction([&](float clock){
			glm::mat4 scale;
			glm::mat4 rotate;
			glm::mat4 translate;
			glm::mat4 model;

			// scale it to "reasonable" world dimensions.
			scale = glm::scale(glm::mat4(), glm::vec3(0.001, 0.001, 0.001));

			// flip it up so it's vertical
			rotate = glm::rotate(glm::mat4(), glm::radians(90.0f), glm::vec3(1.0f, 0.0f, 0.0f));

			// move it so the bottom edge is on the ground plane, and it's centered horizontally.
			translate = glm::translate(glm::mat4(), glm::vec3(-0.96, 2.0, 1.08));

			// applay all those to the modle matrix.
			model = translate * rotate * scale;
			guiCanvas->model = hmdSled->model * model;
		});
	}
	*/

	hmdSled->AddChild(gm);

	// turns out that the "translation" portion of a 4x4 transformation matrix is the x,y,z of row 4
	AddProportionalFunc("LeftThumbStick.x", [](float v) { hmdSled->model[3][0] += v / 10.0f; });
	AddProportionalFunc("LeftThumbStick.y", [](float v) { hmdSled->model[3][1] += v / 10.0f; });
	AddProportionalFunc("LeftIndexTrigger", [](float v) { hmdSled->model[3][2] += v / 10.0f; });
	AddProportionalFunc("LeftHandTrigger",  [](float v) { hmdSled->model[3][2] -= v / 10.0f; });
}