// physx-main.cpp : Defines the entry point for the GLFW-based PhysX application.
// The console window is immediately closed.
#include <PxPhysicsAPI.h>

#include <stdio.h>

#include <glew.h>
#include <GLFW/glfw3.h>

#include <xgl.h>
#include <physx-xgl.h>

static PhysXXGL *pxgl = NULL;

void error_callback(int error, const char *description) {
	fprintf(stderr, "Error: %s\n", description);
}

static void key_callback(GLFWwindow *window, int key, int scancode, int action, int mods) {
	if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS)
		glfwSetWindowShouldClose(window, GLFW_TRUE);
	else {
		if (pxgl != NULL) {
			if (action == GLFW_PRESS)
				pxgl->KeyEvent(key, 0);
			else if (action == GLFW_RELEASE)
				pxgl->KeyEvent(key, 0x8000);
			else if (action == GLFW_REPEAT)
				pxgl->KeyEvent(key, 0x4000);
		}
	}
}

static void cursor_position_callback(GLFWwindow *window, double x, double y) {
	int state = 0;

	state |= glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT);
	state |= glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_RIGHT) << 1;

	if (pxgl != NULL)
		pxgl->MouseEvent((int)x, (int)y, state);
}

int main(void) {
	GLFWwindow *window;
	/*
	if (!FreeConsole()) {
	printf("Freeing the console failed: %d\n", GetLastError());
	exit(0);
	}
	*/
	if (!glfwInit()) {
		printf("glfwInit() failed\n");
		return -1;
	}

	glfwSetErrorCallback(error_callback);

	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 2);
	glfwWindowHint(GLFW_SAMPLES, 8);

	window = glfwCreateWindow(1280, 720, "Mad Style TV Example", NULL, NULL);
	if (!window) {
		printf("glfwCreateWindow() failed\n");
		glfwTerminate();
		return -1;
	}

	glfwMakeContextCurrent(window);

	glfwSetKeyCallback(window, key_callback);
	glfwSetCursorPosCallback(window, cursor_position_callback);
	//glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);

	glfwSwapInterval(1);

	glewExperimental = GL_TRUE;
	GLenum err = glewInit();
	if (GLEW_OK != err) {
		printf("glewInit() failed: %s\n", glewGetErrorString(err));
		exit(-1);
	}

	pathToAssets = "..";

	try {
		pxgl = new PhysXXGL();

		while (!glfwWindowShouldClose(window)) {
			int width, height;
			static int prevWidth = 0, prevHeight = 0;

			glfwGetFramebufferSize(window, &width, &height);
			if ((width != prevWidth) || (height != prevHeight)){
				pxgl->Reshape(width, height);
				prevWidth = width;
				prevHeight = height;
			}

			glClear(GL_COLOR_BUFFER_BIT);

			pxgl->Display();

			glfwSwapBuffers(window);
			glfwPollEvents();
		}

		glfwTerminate();
	}
	catch (std::runtime_error e) {
		printf("Exception: %s\n", e.what());
	}

	return 0;
}
