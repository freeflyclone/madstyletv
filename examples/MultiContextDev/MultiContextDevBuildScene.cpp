/**************************************************************
** MultiContextDevBuildScene.cpp
**
** To support asynchronous image upload for multimedia apps,
** multiple OpenGL contexts are required.  Up until now
** XGL has only had the main OpenGL context.  XGLContext
** provides integration of OpenGL context management for
** multi-threaded OpenGL goodness.
**************************************************************/
#include "ExampleXGL.h"

class XGLContextCanvas : public XGLTexQuad {
public:
	XGLContextCanvas(std::string ip) : XGLTexQuad(ip) {
		xprintf("%s()\n", __FUNCTION__);
	};

	void Draw() {
		xprintf("%s()\n", __FUNCTION__);
		XGLTexQuad::Draw();
	}
};

class XGLContext : public XThread {
public:
	XGLContext(ExampleXGL* p) : pXgl(p), XThread("XGLContextThread") {
		glfwSetErrorCallback(ErrorFunc);
		glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);

		if ((mWindow = glfwCreateWindow(32, 32, "Offscreen", NULL, pXgl->window)) == nullptr)
			xprintf("Oops, glfwCreateWindow() failed\n");
	}

	static void ErrorFunc(int code, const char *str) {
		xprintf("%s(): %d - %s\n", __FUNCTION__, code, str);
	}

	void Run() {
		glfwMakeContextCurrent(mWindow);

		while (IsRunning()) {
			xprintf("%s()\n", __FUNCTION__);
			std::this_thread::sleep_for(std::chrono::duration<int, std::milli>(100));
		}
	}

	GLFWwindow* mWindow;
	ExampleXGL* pXgl;
};

void ExampleXGL::BuildScene() {
	XGLContext *ac = new XGLContext(this);
	XGLContextCanvas *shape;
	std::string imgPath = pathToAssets + "/assets/AndroidDemo.png";

	AddShape("shaders/tex", [&shape, imgPath](){ shape = new XGLContextCanvas(imgPath); return shape; });

	// have the upright texture scaled up and made 16:9 aspect, and orbiting the origin
	// to highlight use of the callback function for animation of a shape.  Note that this function
	// runs once per frame BEFORE the shape's geomentry is rendered.  A lot can
	// be done here. Hint: scripting, physics(?)
	shape->SetAnimationFunction([shape](float clock) {
		float sinFunc = sin(clock / 40.0f) * 10.0f;
		float cosFunc = cos(clock / 40.0f) * 10.0f;
		glm::mat4 scale = glm::scale(glm::mat4(), glm::vec3(9.6f, 5.4f, 1.0f));
		glm::mat4 translate = glm::translate(glm::mat4(), glm::vec3(cosFunc, sinFunc, 5.4f));
		glm::mat4 rotate = glm::rotate(glm::mat4(), glm::radians(90.0f), glm::vec3(1.0f, 0.0f, 0.0f));
		shape->model = translate * rotate * scale;
	});

	ac->Start();
}
