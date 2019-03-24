/**************************************************************
** GameDev01BuildScene.cpp
**
** Simple Game Demo, with PhysX and VR (Oculus) integration.
** This differs from PhysXTest in that PhysX is integrated
** differently, ie: not using a separate main.cpp file.
**************************************************************/
#include "ExampleXGL.h"
#include "XGLPhysX.h"

extern bool initHmd;
XGLPhysX* px;

void ExampleXGL::BuildScene() {
	initHmd = false;

	px = new XGLPhysX(this);

	XInputKeyFunc fireKey = [&](int key, int flags) {
		const bool isDown = (flags & 0x8000) == 0;
		const bool isRepeat = (flags & 0x4000) != 0;
		if (isDown) {
			physx::PxVec3 pos, dir;

			pos.x = camera.pos.x;
			pos.y = camera.pos.y;
			pos.z = camera.pos.z - 2.0f;

			if (hmdSled) {
				glm::vec4 forward = glm::toMat4(hmdSled->o) * glm::vec4(0.0, 1.0, 0.0, 0.0);
				dir.x = forward.x;
				dir.y = forward.y;
				dir.z = forward.z;

			}
			else {
				dir.x = camera.front.x;
				dir.y = camera.front.y;
				dir.z = camera.front.z;
			}

			dir *= 80.0;

			px->createDynamic(physx::PxTransform(pos), physx::PxSphereGeometry(1), dir);
		}
	};
	AddKeyFunc(' ', fireKey);

	// can't use "hmdSled", it gets made AFTER this function is run.
	if (initHmd) {
		// fire forward
		AddProportionalFunc("RightIndexTrigger", [this](float v) {
			// 90Hz is way too fast, slow it down to 10Hz
			if (fmod(clock, 9.0f) == 1.0f) {
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
}
