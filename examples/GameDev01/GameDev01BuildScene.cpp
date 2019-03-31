/**************************************************************
** GameDev01BuildScene.cpp
**
** Simple Game Demo, with PhysX and VR (Oculus) integration.
** This differs from PhysXTest in that PhysX is integrated
** differently, ie: not using a separate main.cpp file.
**************************************************************/
#include "ExampleXGL.h"
#include "XGLPhysX.h"

XGLPhysX* px;
XGLShape *hand = nullptr;
XGLShape *worldCursor = nullptr;

void ExampleXGL::BuildScene() {
	XGLShape *shape;

	// 2 options: set "initHmd" to true, and InitHmd() runs after this method,
	// -or- simply call it here and do NOT set initHmd to true.
	InitHmd();

	hand = (XGLShape*)hmdSled->FindObject("RightHand0");
	if (hand)
		xprintf("Found Right Hand\n");

	px = new XGLPhysX(this);

	const int limit = 40;
	const int step = 10;
	int x = 0, y = 0;

	for (int y = -limit; y <= limit; y += step)	{
		for (int x = -limit; x <= limit; x += step)
		{
			AddShape("shaders/specular", [&]() { shape = new XGLTorus(5.0, 1.0, 54, 32); return shape; });
			shape->attributes.diffuseColor = XGLColors::red;
			shape->attributes.specularColor = XGLColors::white;
			glm::mat4 translate = glm::translate(glm::mat4(), glm::vec3((float)x * 2, (float)y * 2, 1));
			shape->model = translate;
			px->ShapeToActor(shape);
		}
	}

	XInputKeyFunc fireKey = [&](int key, int flags) {
		const bool isDown = (flags & 0x8000) == 0;
		const bool isRepeat = (flags & 0x4000) != 0;
		if (isDown) {
			XPhyPoint p;
			XPhyVelocity v;

			if (hmdSled) {
				p = hmdSled->p + hand->p;
				v = 40.0f * glm::toMat3(hmdSled->o) * glm::toMat3(hand->o) * glm::vec3(0.0, 1.0, 0.0);
			}
			else {
				p = camera.pos;
				v = 40.0f * camera.front;
			}

			px->CreateDynamicSphere(0.3f, p, v);
		}
	};
	AddKeyFunc(' ', fireKey);

	if (hmdSled) {
		// fire forward
		AddProportionalFunc("RightIndexTrigger", [this](float v) {
			// 90Hz is way too fast, slow it down to 10Hz
			if (fmod(clock, 4.5f) == 1.0f) {
				if (v > 0.5f)
					KeyEvent(' ', 0);
				else
					KeyEvent(' ', 0x8000);
			}
		});

		// drop stack of boxes
		AddKeyFunc(1, [this](int key, int flags){
			KeyEvent('B', flags);
		});
	}


	XInputKeyFunc blocksKey = [&](int key, int flags) {
		const bool isDown = (flags & 0x8000) == 0;
		const bool isRepeat = (flags & 0x4000) != 0;
		static float stackY = 0.0f;

		if (isDown && !isRepeat) {
			px->createStack(physx::PxTransform(physx::PxVec3(0, stackY -= 3.0f, 0)), 10, 1.0f);
		}
	};
	AddKeyFunc('B', blocksKey);

	AddShape("shaders/specular", [&]() { worldCursor = new XGLSphere(0.5, 32); return worldCursor; });
	XInputMouseFunc worldCursorMouse = [&](int x, int y, int flags) {
		if (mt.IsTrackingRightButton()) {
			XGLWorldCoord *out = wc.Unproject(projector, x, y);

			px->RayCast(out[0], out[1]);

			// project the worldCursor ray onto the X/Y (Z=0) plane
			// TODO: Figure this out.  I found it on StackOverflow.
			float f = out[0].z / (out[1].z - out[0].z);
			float x2d = out[0].x - f * (out[1].x - out[0].x);
			float y2d = out[0].y - f * (out[1].y - out[0].y);
			float z2d = -0.002f;

			if (worldCursor) {
				glm::mat4 translate = glm::translate(glm::mat4(), glm::vec3(x2d, y2d, z2d));
				worldCursor->model = translate;
			}
		}
		else
			px->ResetActive();
	};
	AddMouseFunc(worldCursorMouse);
}
