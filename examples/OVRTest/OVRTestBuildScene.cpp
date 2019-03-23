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
//XGLSphere *sphere;

const float constSpeed1 = 60.0f * 4.0f;
const float constSpeed2 = 45.0f * 4.0f;
const float constSpeed3 = 30.0f * 4.0f;

float speed1 = constSpeed1;
float speed2 = constSpeed2;
float speed3 = constSpeed3;

// constructed in OVRTestGUI.cpp / ExampleXGL::BuildGui()
extern XGLGuiManager *appGuiManager;

const XGLColor gray = { 0.005, 0.005, 0.005, 1 };

extern bool initHmd;

void ExampleXGL::BuildScene() {
	XGLShape *shape;
	glm::mat4 rotate, translate;

	initHmd = true;

	// create spinny torus thingy...
	if (true) {
		XGLShape *bigGrayTorus;
		AddShape("shaders/specular", [&](){ bigGrayTorus = new XGLTorus(5.0f, 1.0f, 64, 32); return bigGrayTorus; });
		bigGrayTorus->attributes.diffuseColor = gray;
		bigGrayTorus->SetAnimationFunction([bigGrayTorus](float clock) {
			glm::mat4 rotate = glm::rotate(glm::mat4(), clock / speed1, glm::vec3(1.0f, 0.0f, 0.0f));
			bigGrayTorus->model = rotate;
		});

		XGLShape *bgtChildTransformer;
		CreateShape("shaders/specular", [&](){ bgtChildTransformer = new XGLTransformer(); return bgtChildTransformer; });
		bgtChildTransformer->SetAnimationFunction([bgtChildTransformer](float clock) {
			float translateFunction = sin(clock / 180.0f);
			glm::mat4 rotate = glm::rotate(glm::mat4(), clock / speed2, glm::vec3(0.0f, 0.0f, 1.0f));
			bgtChildTransformer->model = rotate;
		});
		bigGrayTorus->AddChild(bgtChildTransformer);

		XGLShape *bgtChildRedTorus;
		CreateShape("shaders/specular", [&](){ bgtChildRedTorus = new XGLTorus(2.0f, 0.5f, 64, 32); return bgtChildRedTorus; });
		bgtChildRedTorus->attributes.diffuseColor = XGLColors::red;
		translate = glm::translate(glm::mat4(), glm::vec3(5.0, 0, 0));
		rotate = glm::rotate(glm::mat4(), glm::radians(90.0f), glm::vec3(1.0f, 0.0f, 0.0f));
		bgtChildRedTorus->model = translate * rotate;
		bgtChildTransformer->AddChild(bgtChildRedTorus);

		XGLShape *bgtCrtTransformer;
		CreateShape("shaders/000-simple", [&](){ bgtCrtTransformer = new XGLTransformer(); return bgtCrtTransformer; });
		bgtCrtTransformer->SetAnimationFunction([bgtCrtTransformer](float clock) {
			float translateFunction = sin(clock / 180.0f);
			glm::mat4 rotate = glm::rotate(glm::mat4(), clock / speed3, glm::vec3(0.0f, 0.0f, 1.0f));
			bgtCrtTransformer->model = rotate;
		});
		bgtChildRedTorus->AddChild(bgtCrtTransformer);

		XGLShape *bgtGrandChildYellowTorus;
		CreateShape("shaders/specular", [&](){ bgtGrandChildYellowTorus = new XGLTorus(0.75f, 0.25f, 64, 32); return bgtGrandChildYellowTorus; });
		bgtGrandChildYellowTorus->attributes.diffuseColor = XGLColors::yellow;
		translate = glm::translate(glm::mat4(), glm::vec3(2.0, 0, 0));
		rotate = glm::rotate(glm::mat4(), glm::radians(90.0f), glm::vec3(1.0f, 0.0f, 0.0f));
		bgtGrandChildYellowTorus->model = translate * rotate;
		bgtCrtTransformer->AddChild(bgtGrandChildYellowTorus);
	}

	AddShape("shaders/specular", [&](){ shape = new XGLTorus(3.0f, 0.5f, 64, 32); return shape; });
	shape->attributes.diffuseColor = XGLColors::blue;
	shape->model = glm::translate(glm::mat4(), glm::vec3(20, 0, 0));

	AddShape("shaders/specular", [&](){ shape = new XGLTorus(3.0f, 0.5f, 64, 32); return shape; });
	shape->attributes.diffuseColor = XGLColors::red;
	shape->model = glm::translate(glm::mat4(), glm::vec3(-20, 0, 0)) * glm::scale(glm::mat4(), glm::vec3(2, 2, 2));

	AddShape("shaders/specular", [&](){ shape = new XGLTorus(3.0f, 0.5f, 64, 32); return shape; });
	shape->attributes.diffuseColor = XGLColors::green;
	shape->model = glm::translate(glm::mat4(), glm::vec3(30, 0, 0))
		* glm::rotate(glm::mat4(), glm::radians(90.0f), glm::vec3(0, 1, 0))
		* glm::scale(glm::mat4(), glm::vec3(2, 2, 2));

	// function to toggle wire frame rendering
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

	// add the function to M key handler, both upper and lower case.
	AddKeyFunc('M', renderMod);
	AddKeyFunc('m', renderMod);

	XInputKeyFunc toggleHud = [&](int key, int flags) {
		const bool isDown = (flags & 0x8000) == 0;
		const bool isRepeat = (flags & 0x4000) != 0;
		static bool isHudVisible = false;

		if (isDown && !isRepeat){
			isHudVisible = isHudVisible ? false : true;
			appGuiManager->isVisible = isHudVisible;
		}
	};
	AddKeyFunc('h', toggleHud);
}