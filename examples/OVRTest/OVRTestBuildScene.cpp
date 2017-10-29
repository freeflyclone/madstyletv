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
XGLSled *hmdSled;
XGLShape *rightFinger, *rightThumb, *leftFinger, *leftThumb;

const float constSpeed1 = 60.0f * 4.0f;
const float constSpeed2 = 45.0f * 4.0f;
const float constSpeed3 = 30.0f * 4.0f;

float speed1 = constSpeed1;
float speed2 = constSpeed2;
float speed3 = constSpeed3;

// constructed in OVRTestGUI.cpp / ExampleXGL::BuildGui()
extern XGLGuiManager *appGuiManager;

const XGLColor gray = { 0.005, 0.005, 0.005, 1 };

void ExampleXGL::BuildScene() {
	XGLShape *shape;
	glm::mat4 rotate, translate;

	// create spinny torus thingy...
	if (true) {
		XGLShape *bigGrayTorus;
		CreateShape("shaders/specular", [&](){ bigGrayTorus = new XGLTorus(5.0f, 1.0f, 64, 32); return bigGrayTorus; });
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
		rootShape->AddChild(bigGrayTorus);
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

	CreateShape("shaders/specular", [&](){ shape = new XGLTorus(3.0f, 0.5f, 64, 32); return shape; });
	shape->attributes.diffuseColor = XGLColors::blue;
	shape->model = glm::translate(glm::mat4(), glm::vec3(20, 0, 0));
	rootShape->AddChild(shape);

	CreateShape("shaders/specular", [&](){ shape = new XGLTorus(3.0f, 0.5f, 64, 32); return shape; });
	shape->attributes.diffuseColor = XGLColors::red;
	shape->model = glm::translate(glm::mat4(), glm::vec3(-20, 0, 0)) * glm::scale(glm::mat4(), glm::vec3(2, 2, 2));
	rootShape->AddChild(shape);

	CreateShape("shaders/specular", [&](){ shape = new XGLTorus(3.0f, 0.5f, 64, 32); return shape; });
	shape->attributes.diffuseColor = XGLColors::green;
	shape->model = glm::translate(glm::mat4(), glm::vec3(30, 0, 0))
		* glm::rotate(glm::mat4(), glm::radians(90.0f), glm::vec3(0, 1, 0))
		* glm::scale(glm::mat4(), glm::vec3(2, 2, 2));
	rootShape->AddChild(shape);

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
	AddKeyFunc('~', toggleHud);

	// LeftHand, anchored to the viewpoint
	AddShape("shaders/specular", [&]() { shape = new XGLSphere(0.05f, 64); shape->SetName("LeftHand"); return shape; });

	// Left Finger, child of LeftHand
	CreateShape("shaders/specular", [&]() { leftFinger = new XGLCapsule(0.01f, 0.1f, 32); leftFinger->SetName("LeftFinger"); return leftFinger; });
	translate = glm::translate(glm::mat4(), glm::vec3(0.0, 0.0, -0.1));
	rotate = glm::rotate(glm::mat4(), glm::radians(90.0f), glm::vec3(0.0, 1.0, 0.0));
	leftFinger->model = translate * rotate;
	shape->AddChild(leftFinger);

	// LeftThumb, child of LeftHand
	CreateShape("shaders/specular", [&]() { leftThumb = new XGLCapsule(0.01f, 0.075f, 32); leftThumb->SetName("LeftThumb"); return leftThumb; });
	translate = glm::translate(glm::mat4(), glm::vec3(0.06, 0.0, -0.06));
	rotate = glm::rotate(glm::mat4(), glm::radians(45.0f), glm::vec3(0.0, 1.0, 0.0));
	leftThumb->model = translate * rotate;
	shape->AddChild(leftThumb);


	// RightHand, anchored to the viewpoint
	AddShape("shaders/specular", [&]() { shape = new XGLSphere(0.05f, 64); shape->SetName("RightHand"); return shape; });

	// RightFinger, child of RightHand
	CreateShape("shaders/specular", [&]() { rightFinger = new XGLCapsule(0.01f, 0.1f, 32); rightFinger->SetName("RightFinger"); return rightFinger; });
	translate = glm::translate(glm::mat4(), glm::vec3(0.0, 0.0, -0.1));
	rotate = glm::rotate(glm::mat4(), glm::radians(90.0f), glm::vec3(0.0, 1.0, 0.0));
	rightFinger->model = translate * rotate;
	shape->AddChild(rightFinger);

	// RightThumb, child of Right Hand
	CreateShape("shaders/specular", [&]() { rightThumb = new XGLCapsule(0.01f, 0.075f, 32); rightThumb->SetName("RightThumb"); return rightThumb; });
	translate = glm::translate(glm::mat4(), glm::vec3(-0.06, 0.0, -0.06));
	rotate = glm::rotate(glm::mat4(), glm::radians(135.0f), glm::vec3(0.0, 1.0, 0.0));
	rightThumb->model = translate * rotate;
	shape->AddChild(rightThumb);

	// move forward
	AddProportionalFunc("LeftIndexTrigger", [this](float v) { 
		glm::vec4 forward = glm::vec4(0.0, -v / 10.0f, 0.0, 0.0) * glm::toMat4(rootShape->o);
		rootShape->p += glm::vec3(forward);

		rootShape->model = rootShape->GetFinalMatrix(); 
	});

	// move backward
	AddProportionalFunc("LeftHandTrigger", [this](float v) {
		glm::vec4 backward = glm::vec4(0.0, v / 10.0f, 0.0, 0.0) * glm::toMat4(rootShape->o);
		rootShape->p += glm::vec3(backward);

		rootShape->model = rootShape->GetFinalMatrix();
	});

	// yaw (rudder)
	AddProportionalFunc("LeftThumbStick.x", [this](float v) { rootShape->SampleInput(v, 0.0f, 0.0f); });

	// pitch (elevator)
	AddProportionalFunc("RightThumbStick.y", [this](float v) { rootShape->SampleInput(0.0f, v, 0.0f); });

	// roll (ailerons)
	AddProportionalFunc("RightThumbStick.x", [this](float v) { rootShape->SampleInput(0.0f, 0.0f, -v); });
}