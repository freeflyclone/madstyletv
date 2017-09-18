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
XGLShape *hmdSled;
XGLShape *rightFinger, *rightThumb, *leftFinger, *leftThumb;

const float constSpeed1 = 60.0f * 4.0f;
const float constSpeed2 = 45.0f * 4.0f;
const float constSpeed3 = 30.0f * 4.0f;

float speed1 = constSpeed1;
float speed2 = constSpeed2;
float speed3 = constSpeed3;

// constructed in OVRTestGUI.cpp / ExampleXGL::BuildGui()
extern XGLGuiManager *appGuiManager;

void ExampleXGL::BuildScene() {
	XGLShape *shape, *child1, *child2, *child3, *child4;
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
	hmdSled->SetAnimationFunction([&](float clock) {
		glm::mat4 t = glm::translate(glm::mat4(), hmdSled->p);
		hmdSled->model = t;
	});

	CreateShape("shaders/specular", [&](){ shape = new XGLSphere(0.01f, 4); shape->SetName("SledOrigin");  return shape; }, 2);
	hmdSled->AddChild(shape);

	CreateShape("shaders/specular", [&]() { shape = new XGLSphere(0.05f, 64); shape->SetName("LeftHand"); return shape; });
	hmdSled->AddChild(shape);

	CreateShape("shaders/specular", [&]() { leftFinger = new XGLCapsule(0.01f, 0.1, 32); leftFinger->SetName("LeftFinger"); return leftFinger; });
	translate = glm::translate(glm::mat4(), glm::vec3(0.0, 0.0, -0.1));
	rotate = glm::rotate(glm::mat4(), glm::radians(90.0f), glm::vec3(0.0, 1.0, 0.0));
	leftFinger->model = translate * rotate;
	shape->AddChild(leftFinger);

	CreateShape("shaders/specular", [&]() { leftThumb = new XGLCapsule(0.01f, 0.075, 32); leftThumb->SetName("RightThumb"); return leftThumb; });
	translate = glm::translate(glm::mat4(), glm::vec3(0.06, 0.0, -0.06));
	rotate = glm::rotate(glm::mat4(), glm::radians(45.0f), glm::vec3(0.0, 1.0, 0.0));
	leftThumb->model = translate * rotate;
	shape->AddChild(leftThumb);

	CreateShape("shaders/specular", [&]() { shape = new XGLSphere(0.05f, 64); shape->SetName("RightHand"); return shape; });
	hmdSled->AddChild(shape);

	CreateShape("shaders/specular", [&]() { rightFinger = new XGLCapsule(0.01f, 0.1, 32); rightFinger->SetName("RightFinger"); return rightFinger; });
	translate = glm::translate(glm::mat4(), glm::vec3(0.0, 0.0, -0.1));
	rotate = glm::rotate(glm::mat4(), glm::radians(90.0f), glm::vec3(0.0, 1.0, 0.0));
	rightFinger->model = translate * rotate;
	shape->AddChild(rightFinger);

	CreateShape("shaders/specular", [&]() { rightThumb = new XGLCapsule(0.01f, 0.075, 32); rightThumb->SetName("RightThumb"); return rightThumb; });
	translate = glm::translate(glm::mat4(), glm::vec3(-0.06, 0.0, -0.06));
	rotate = glm::rotate(glm::mat4(), glm::radians(135.0f), glm::vec3(0.0, 1.0, 0.0));
	rightThumb->model = translate * rotate;
	shape->AddChild(rightThumb);

	hmdSled->AddChild(appGuiManager);

	AddProportionalFunc("LeftThumbStick.x", [](float v) { hmdSled->p.x += v / 10.0f; });
	AddProportionalFunc("LeftThumbStick.y", [](float v) { hmdSled->p.y += v / 10.0f; });
	AddProportionalFunc("LeftIndexTrigger", [](float v) { hmdSled->p.z += v / 10.0f; });
	AddProportionalFunc("LeftHandTrigger",  [](float v) { hmdSled->p.z -= v / 10.0f; });
}