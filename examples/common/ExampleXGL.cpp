#include "ExampleXGL.h"

// TODO:  I don't think I need to initialize "wc" this way if I'm using
// lambda functions for the world cursor.  Will investigate.
ExampleXGL::ExampleXGL() : wc(&shaderMatrix) {
	XGLShape *shape;

	// add 2D shapes to the guiShapes list.
	BuildGUI();

	// Initialize the Camera matrix
	glm::vec3 cameraPosition(-15, -25, 15);
	glm::vec3 cameraDirection = glm::normalize(cameraPosition*-1.0f);
	glm::vec3 cameraUp = { 0, 0, 1 };
	camera.Set(cameraPosition, cameraDirection,	cameraUp);

	// set a camera animation function
	camera.SetTheFunk([this](XGLCamera *c){
		glm::float32 speed = 0.6f;

		if (kt.f)
			c->pos += glm::normalize(c->front) * speed;
		if (kt.b)
			c->pos -= glm::normalize(c->front) * speed;
		if (kt.r)
			c->pos += glm::normalize(glm::cross(c->front, c->up)) * speed;
		if (kt.l)
			c->pos -= glm::normalize(glm::cross(c->front, c->up)) * speed;

		if (mt.IsTrackingLeftButton() && !mt.IsTrackingRightButton()) {
			glm::float32 yaw = 0.0f;
			glm::float32 pitch = 0.0f;

			if (mt.dx || mt.dy) {
				// get mouse tracker x,y into camera orientation
				yaw = ((glm::float32) - mt.dx) * 0.001f;
				pitch = ((glm::float32)mt.dy) * 0.001f;

				// process yaw first
				glm::mat4 cameraYaw = glm::rotate(glm::mat4(), yaw, camera.up);
				glm::vec4 front = cameraYaw * glm::vec4(camera.front, 1.0f);
				camera.front = glm::vec3(front.x, front.y, front.z);

				// pitch involves rotating around the "right" vector, which doesn't exist, so make one from front & up
				glm::mat4 cameraPitch = glm::rotate(glm::mat4(), pitch, glm::cross(camera.up, camera.front));
				front = cameraPitch * glm::vec4(camera.front, 1.0f);

				// avoid  gimbal lock: absolute dot product (magnitude of parallelness) of front vs up
				glm::float32 pitchVerticality = glm::abs(glm::dot(glm::vec3(front.x, front.y, front.z), camera.up));

				// enforce limit (emperically derived value)
				if (pitchVerticality < 0.99f) {
					camera.front = glm::vec3(front.x, front.y, front.z);
				}

				// clear mouse tracker delta position
				mt.Done();
			}
		}
	});

	// add mouse event handling (XInput class) function mapping
	AddMouseFunc([this](int x, int y, int flags){
		if (GuiIsActive())
			GuiResolveMouseEvent(GetGuiManager(), x, y, flags);
		else
			mt.Event(x, y, flags);
	});

	// add key event handling (XInput class) function mapping
	AddKeyFunc(std::make_pair('A', 'Z'), [this](int key, int flags) {
		const bool isDown = (flags & 0x8000) == 0;
		const bool isRepeat = (flags & 0x4000) != 0;

		if (isDown) {
			switch (key) {
			case 'W': kt.f = true; break;
			case 'S': kt.b = true; break;
			case 'A': kt.l = true; break;
			case 'D': kt.r = true; break;
			case '+': kt.wu = true; break;
			case '-': kt.wd = true; break;
			}
		}
		else {
			switch (key) {
			case 'W': kt.f = false; break;
			case 'S': kt.b = false; break;
			case 'A': kt.l = false; break;
			case 'D': kt.r = false; break;
			case '+': kt.wu = false; break;
			case '-': kt.wd = false; break;
			}
		}
	});

	// add a default "ground" plane grid.
	AddShape("shaders/000-simple", [&](){ shape = new XYPlaneGrid(); return shape; });

	// Features of the framework are incrementally introduced by enhancing this function
	// on a per example basis.
	BuildScene();

	// set the following to 'true' to enable Oculus Rift with cockpit flight controls on Touch Controllers.
	if (true) {
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

		AddProportionalFunc("LeftThumbStick.y", [this](float v) {
			glm::vec4 backward = glm::toMat4(hmdSled->o) * glm::vec4(0.0, v / 10.0f, 0.0, 0.0);
			hmdSled->p += glm::vec3(backward);
			hmdSled->model = hmdSled->GetFinalMatrix();
		});

		// yaw (rudder)
		AddProportionalFunc("LeftThumbStick.x", [this](float v) { hmdSled->SampleInput(-v, 0.0f, 0.0f); });

		// pitch (elevator)
		AddProportionalFunc("RightThumbStick.y", [this](float v) { hmdSled->SampleInput(0.0f, -v, 0.0f); });

		// roll (ailerons)
		AddProportionalFunc("RightThumbStick.x", [this](float v) { hmdSled->SampleInput(0.0f, 0.0f, v); });

		// change the default configuration so the HMD will work.
		preferredWidth = 1080;
		preferredHeight = 600;

		pHmd = new XGLHmd(this, preferredWidth, preferredHeight);
		useHmd = true;
		preferredSwapInterval = 0;
	}
}

void ExampleXGL::Reshape(int w, int h) {
	try {
		width = w;
		height = h;
		projector.Reshape(w, h);
		Display();
	}
	catch (std::runtime_error e){
		xprintf("Well that didn't work: %s\n", e.what());
	}
}

bool ExampleXGL::Display() {
	if (pHmd)
		return pHmd->Loop();
	else
		return XGL::Display();
}

ExampleXGL::~ExampleXGL() {
	if (pHmd)
		delete pHmd;
}
