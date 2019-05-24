/**************************************************************
** MultiContextDevBuildScene.cpp
**
** To support asynchronous image upload for multimedia apps,
** multiple OpenGL contexts are required.  Up until now
** XGL has only had the main OpenGL context.  XGLContext
** provides integration of OpenGL context management for
** multi-threaded OpenGL goodness.
**
** Start with a background thread / GL context that repeatedly 
** copies alternating frames to a PBO to see what opengl debug
** output has to say about buffer usage.
**************************************************************/
#include "ExampleXGL.h"

class XGLContext : public XThread {
public:
	XGLContext(ExampleXGL* pxgl, int w, int h, int c) : pXgl(pxgl), width(w), height(h), channels(c), XThread("XGLContextThread") {
		glfwSetErrorCallback(ErrorFunc);
		glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);

		if ((mWindow = glfwCreateWindow(32, 32, "Offscreen", NULL, pXgl->window)) == nullptr)
			xprintf("Oops, glfwCreateWindow() failed\n");

		glfwMakeContextCurrent(mWindow);

		pboSize = width*height*channels;
		glGenBuffers(1, &pboId);
		GL_CHECK("glGenBuffers failed");

		glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pboId);
		GL_CHECK("glBindBuffer() failed");

		glBufferData(GL_PIXEL_UNPACK_BUFFER, pboSize, nullptr, GL_STREAM_DRAW);
		GL_CHECK("glBufferData() failed");

		pboBuffer = (uint8_t*)glMapBuffer(GL_PIXEL_UNPACK_BUFFER, GL_WRITE_ONLY);
		GL_CHECK("glMapBuffer() failed");

		if (pboBuffer == nullptr) {
			xprintf("Doh! glMapBuffer() returned nullptr\n");
		}

		glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
		GL_CHECK("glBindBuffer() failed");

		glfwMakeContextCurrent(pXgl->window);

		black = new uint8_t[pboSize];
		memset(black, 0, pboSize);

		white = new uint8_t[pboSize];
		memset(white, 255, pboSize);
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
	GLuint pboId;
	uint8_t* pboBuffer;
	int width, height, channels;
	int pboSize;

	uint8_t *black, *white;
};

void ExampleXGL::BuildScene() {
	XGLContext *ac = new XGLContext(this, 1920,1088,3);
	XGLTexQuad *shape;
	std::string imgPath = pathToAssets + "/assets/AndroidDemo.png";

	AddShape("shaders/tex", [&shape, imgPath](){ shape = new XGLTexQuad(imgPath); return shape; });

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
