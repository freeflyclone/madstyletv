// main.cpp : Defines the entry point for the GLFW application.
// The console window is immediately closed.
#include <Windows.h>
#include <stdio.h>

#include <glew.h>
#include <GLFW/glfw3.h>

#include <xgl.h>
#include <ExampleXGL.h>

static ExampleXGL *exgl = NULL;

void SetGlobalWorkingDirectoryName()
{
	DWORD sizeNeeded = GetCurrentDirectory(0, NULL);
	DWORD size;
	TCHAR *buff = new TCHAR[sizeNeeded];

	if ((size = GetCurrentDirectory(sizeNeeded, buff)) != sizeNeeded - 1)
		throwXGLException("GetCurrentDirectory() unexpectedly failed. " + std::to_string(size) + " vs " + std::to_string(sizeNeeded));

#ifdef UNICODE
	std::wstring wstr(buff);
	currentWorkingDir = std::string(wstr.begin(), wstr.end());
#else
	currentWorkingDir = std::string(buff);
#endif
	delete[] buff;

	// this presumes that a VS Post-Build step copies the result of the build to $(SolutionDir)\bin
	// it is further presumed that ALL OS specific build procedures will do the same.
	// (also works if project's Debug property "working directory" is set to $(SolutionDir))
	// So we got that going for us.
	pathToAssets = currentWorkingDir.substr(0, currentWorkingDir.rfind("\\bin"));
}

void error_callback(int error, const char *description) {
	fprintf(stderr, "Error: %s\n", description);
}

static void key_callback(GLFWwindow *window, int key, int scancode, int action, int mods) {
	if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS)
		glfwSetWindowShouldClose(window, GLFW_TRUE);
	else {
		if (exgl != NULL) {
			if (action == GLFW_PRESS)
				exgl->KeyEvent(key, 0);
			else if (action == GLFW_RELEASE)
				exgl->KeyEvent(key, 0x8000);
		}
	}
}

static void cursor_position_callback(GLFWwindow *window, double x, double y) {
	int state = 0;
	
	state |= glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT);
	state |= glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_RIGHT) << 1;

	if (exgl != NULL) 
		exgl->MouseEvent((int)x, (int)y, state);
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
	
	SetGlobalWorkingDirectoryName();

	xprintf("current working dir: %s\n", currentWorkingDir.c_str());
	pathToAssets = currentWorkingDir + "\\..";

	xprintf("pathToAssets: %s\n", pathToAssets.c_str());

	try {
		exgl = new ExampleXGL();

		while (!glfwWindowShouldClose(window)) {
			int width, height;

			glfwPollEvents();
			glfwGetFramebufferSize(window, &width, &height);
			exgl->Reshape(width, height);

			glfwSwapBuffers(window);

			glClear(GL_COLOR_BUFFER_BIT);

			exgl->Display();
		}

		glfwTerminate();
	}
	catch (std::runtime_error e) {
		printf("Exception: %s\n", e.what());
	}

	return 0;
}
