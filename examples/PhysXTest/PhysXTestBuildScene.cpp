/**************************************************************
** PhysXTestBuildScene.cpp
**
** Placeholder for PhysX integration testing.
**************************************************************/
#include "physx-xgl.h"

XGLSphere *sphere;
XGLSled *hmdSled;

void ExampleXGL::BuildScene() {
	// Create a cockpit that can be flown in the world, put it in layer 2 to override world object rendering
	// (Turns out the layers hack only works between top level shapes right now)
	AddShape("shaders/000-simple", [&]() { hmdSled = new XGLSled(); return hmdSled; }, 2);
	hmdSled->SetName("HmdSled", false);

	// move forward
	AddProportionalFunc("LeftIndexTrigger", [this](float v) {
		glm::vec4 forward = glm::toMat4(hmdSled->o) * glm::vec4(0.0, v / 10.0f, 0.0, 0.0);
		hmdSled->p += glm::vec3(forward);
		hmdSled->model = hmdSled->GetFinalMatrix();
	});

	// move backward
	AddProportionalFunc("LeftHandTrigger", [this](float v) {
		glm::vec4 backward = glm::toMat4(hmdSled->o) * glm::vec4(0.0, -v / 10.0f, 0.0, 0.0);
		hmdSled->p += glm::vec3(backward);
		hmdSled->model = hmdSled->GetFinalMatrix();
	});

	// yaw (rudder)
	AddProportionalFunc("LeftThumbStick.x", [this](float v) { hmdSled->SampleInput(-v, 0.0f, 0.0f); });

	// pitch (elevator)
	AddProportionalFunc("RightThumbStick.y", [this](float v) { hmdSled->SampleInput(0.0f, -v, 0.0f); });

	// roll (ailerons)
	AddProportionalFunc("RightThumbStick.x", [this](float v) { hmdSled->SampleInput(0.0f, 0.0f, v); });
}

void PhysXXGL::BuildScene() {

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

	XInputKeyFunc fireKey = [&](int key, int flags) {
		const bool isDown = (flags & 0x8000) == 0;
		const bool isRepeat = (flags & 0x4000) != 0;
		if (isDown) {
			physx::PxVec3 pos, dir;

			pos.x = camera.pos.x;
			pos.y = camera.pos.y;
			pos.z = camera.pos.z - 2.0f;

			dir.x = camera.front.x;
			dir.y = camera.front.y;
			dir.z = camera.front.z;

			dir *= 80.0;

			createDynamic(physx::PxTransform(pos), physx::PxSphereGeometry(1), dir);
		}
	};
	AddKeyFunc(' ', fireKey);

	XInputKeyFunc blocksKey = [&](int key, int flags) {
		const bool isDown = (flags & 0x8000) == 0;
		const bool isRepeat = (flags & 0x4000) != 0;
		static float stackY = 0.0f;

		if (isDown && !isRepeat) {
			createStack(physx::PxTransform(physx::PxVec3(0, stackY -= 3.0f, 0)), 10, 1.0f);
		}
	};
	AddKeyFunc('B', blocksKey);

	AddShape("shaders/specular", [&]() { sphere = new XGLSphere(0.5, 32); return sphere; });

	XInputMouseFunc worldCursorMouse = [&](int x, int y, int flags) {
		if (mt.IsTrackingRightButton()) {
			XGLWorldCoord *out = wc.Unproject(projector, x, y);

			RayCast(out[1], out[0]);

			// project the worldCursor ray onto the X/Y (Z=0) plane
			// TODO: Figure this out.  I found it on StackOverflow.
			float f = out[0].z / (out[1].z - out[0].z);
			float x2d = out[0].x - f * (out[1].x - out[0].x);
			float y2d = out[0].y - f * (out[1].y - out[0].y);
			float z2d = -0.002f;

			if (sphere) {
				glm::mat4 translate = glm::translate(glm::mat4(), glm::vec3(x2d, y2d, z2d));
				sphere->model = translate;
			}
		}
		else
			ResetActive();
	};
	AddMouseFunc(worldCursorMouse);
}
