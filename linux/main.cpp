#include <stdio.h>

#include <glew.h>
#include <GLFW/glfw3.h>

#include <xgl.h>
#include <ExampleXGL.h>

static ExampleXGL *exgl = NULL;

void error_callback(int error, const char *description) {
	fprintf(stderr, "Error: %s\n", description);
}

static void key_callback(GLFWwindow *window, int key, int scancode, int action, int mods) {
	if (key==GLFW_KEY_ESCAPE && action==GLFW_PRESS)
		glfwSetWindowShouldClose(window, GLFW_TRUE);
	else {
		if (exgl!=NULL) {
			if (action==GLFW_PRESS)
				exgl->KeyEvent(key, 0);
			else if (action==GLFW_RELEASE)
				exgl->KeyEvent(key, 0x8000);
		}
	}
}

static void cursor_position_callback(GLFWwindow *window, double x, double y) {
	int state = glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT);
	if (state==GLFW_PRESS)
		exgl->MouseEvent( (int)x, (int)y, 1);
}

int main(void) {
	GLFWwindow *window;

	if(!glfwInit()) {
		printf("glfwInit() failed\n");
		return -1;
	}

	glfwSetErrorCallback(error_callback);

	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
	glfwWindowHint(GLFW_SAMPLES, 8);

	window = glfwCreateWindow(1920, 1080, "Mad Style TV Example", NULL, NULL);
	if (!window) {
		printf("glfwCreateWindow() failed\n");
		glfwTerminate();
		return -1;
	}	

	glfwMakeContextCurrent(window);

	glfwSetKeyCallback(window, key_callback);
	glfwSetCursorPosCallback(window, cursor_position_callback);
	glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);

	glfwSwapInterval(1);

	glewExperimental = GL_TRUE;
	GLenum err = glewInit();
	if (GLEW_OK != err) {
		printf("glewInit() failed: %s\n", glewGetErrorString(err));
		exit(-1);
	}	

	pathToAssets = "..";
	exgl = new ExampleXGL();

	while (!glfwWindowShouldClose(window)) {
		int width, height;

		glfwGetFramebufferSize(window, &width, &height);
		exgl->Reshape(width, height);

		glClear(GL_COLOR_BUFFER_BIT);

		exgl->Display();

		glfwSwapBuffers(window);
		glfwPollEvents();
	}

	glfwTerminate();

	return 0;
}
