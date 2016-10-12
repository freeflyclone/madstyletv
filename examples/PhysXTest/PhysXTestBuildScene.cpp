/**************************************************************
** PhysXTestBuildScene.cpp
**
** Placeholder for PhysX integration testing.
**************************************************************/
#include "physx-xgl.h"

void ExampleXGL::BuildScene() {}

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

	XInputMouseFunc mouseFunc = [&](int x, int y, int flags) {
	};
	AddMouseFunc(mouseFunc);
}
