#include <glew.h>
#include <GLFW/glfw3.h>
#include <GL/gl.h>
#include <stdio.h>

#include <xgl.h>

class ExampleXGL : public XGL {
public:
	ExampleXGL() { printf("ExampleXGL::ExampleXGL()\n"); }
	virtual ~ExampleXGL() { printf("~ExampleXGL::ExampleXGL()\n"); }
	virtual void Display() {};
};

int main(void) {
	GLFWwindow *window;

	if(!glfwInit()) {
		printf("glfwInit() failed\n");
		return -1;
	}

	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
	//glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
	//glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

	window = glfwCreateWindow(640, 480, "Hello World", NULL, NULL);
	if (!window) {
		printf("glfwCreateWindow() failed\n");
		glfwTerminate();
		return -1;
	}	

	glfwMakeContextCurrent(window);

	glewExperimental = GL_TRUE;
	GLenum err = glewInit();
	if (GLEW_OK != err) {
		printf("glewInit() failed: %s\n", glewGetErrorString(err));
		exit(-1);
	}	

	ExampleXGL *exgl = new ExampleXGL();

	while (!glfwWindowShouldClose(window)) {
		//glClear(GL_COLOR_BUFFER_BIT);
		glfwSwapBuffers(window);
		glfwPollEvents();
	}

	glfwTerminate();

	return 0;
}
