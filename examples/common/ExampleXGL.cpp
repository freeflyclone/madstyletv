#include "ExampleXGL.h"

// change to global from const, so it can be changed on a per-project basis
bool initHmd = false;

// to allow XGL to create secondary OpenGL contexts, we need ptr to "parent"
// context, which comes from GLFWwindow pointer.
//
// Additional constructor so as not to break WinMain() based ExampleXGL projects.
// Although those could be deprecated and I wouldn't care.
ExampleXGL::ExampleXGL(GLFWwindow* w) : wc(&shaderMatrix) {
	Py_Initialize();
	Initialize(w);
}

ExampleXGL::ExampleXGL() : wc(&shaderMatrix) {
	Initialize(nullptr);
}

// TODO:  I don't think I need to initialize "wc" this way if I'm using
// lambda functions for the world cursor.  Will investigate.
void ExampleXGL::Initialize(GLFWwindow *w) {
	XGLShape *shape;

	window = w;

	// Build a 2D UI per application.  Multiple UI frameworks are available.
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
		if (GuiIsActive()) {
			XGLGuiManager *gm = GetGuiManager();
			if (gm)
				GuiResolveMouseEvent(gm, x, y, flags);
		} 
		else {
			mt.Event(x, y, flags);
		}
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

	// add a default "ground" plane grid.
	AddShape("shaders/000-simple", [&](){ shape = new XYPlaneGrid(); return shape; });

	// Features of the framework are incrementally introduced by enhancing this function
	// on a per example basis.
	BuildScene();

	// set the following to 'true' to enable Oculus Rift with cockpit flight controls on Touch Controllers.
	if (initHmd)
		InitHmd();
}

void ExampleXGL::Reshape(int w, int h) {
	try {
		width = (w<=0)?1:w;
		height = (h<=0)?1:h;

		projector.Reshape(width, height);
		//Display();
	}
	catch (std::runtime_error e){
		xprintf("Well that didn't work: %s\n", e.what());
	}
}

bool ExampleXGL::Display() {
#ifndef LINUX
	if (pHmd)
		return pHmd->Loop();
	else
#endif
		return XGL::Display();
}

ExampleXGL::~ExampleXGL() {
	if (pHmd)
		delete pHmd;
	Py_Finalize();
}
