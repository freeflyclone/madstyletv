// main.cpp : Defines the entry point for the GLFW application.
// The console window is immediately closed.
#ifdef _WIN32
#include <Windows.h>
#else
#include <unistd.h>
#endif
#include <stdio.h>

#include <glew.h>
#include <GLFW/glfw3.h>

#include <xgl.h>
#include <ExampleXGL.h>

static ExampleXGL *exgl = NULL;
#ifndef OPENGL_MAJOR_VERSION
#define OPENGL_MAJOR_VERSION 4
#endif

#ifndef OPENGL_MINOR_VERSION
#define OPENGL_MINOR_VERSION 5
#endif

#ifndef GLFW_WINDOW_TITLE
#define GLFW_WINDOW_TITLE "Mad Style TV Example"
#endif

#ifdef _WIN32
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

#else
void SetGlobalWorkingDirectoryName() {
	char buff[FILENAME_MAX];
	char *xclass_dir = getenv("XCLASS_DIR");

	if (xclass_dir) {
		currentWorkingDir = std::string(xclass_dir);
	}
	else {
		getcwd(buff, sizeof(buff));
		currentWorkingDir = std::string(buff);
	}
	xprintf("Cwd: %s\n", currentWorkingDir.c_str());
}
#endif

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

static void mouse_button_callback(GLFWwindow *window, int button, int action, int mods) {
	int state = 0;
	double x, y;

	glfwGetCursorPos(window, &x, &y);
	state |= glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT);
	state |= glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_RIGHT) << 1;

	if (exgl != NULL)
		exgl->MouseEvent((int)x, (int)y, state);
}

static void window_size_callback(GLFWwindow *window, int width, int height) {
	if (exgl != NULL) {
		exgl->Reshape(width, height);
		exgl->Display();
		glfwSwapBuffers(window);
	}
}

static void window_refresh_callback(GLFWwindow *window){
	if (exgl != NULL)
		exgl->Display();
}

int main(void) {
	GLFWwindow *window;
	int width, height;
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

	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, OPENGL_MAJOR_VERSION);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, OPENGL_MINOR_VERSION);
	glfwWindowHint(GLFW_SAMPLES, 8);

	window = glfwCreateWindow(1280, 720, GLFW_WINDOW_TITLE, NULL, NULL);
	if (!window) {
		printf("glfwCreateWindow() failed\n");
		glfwTerminate();
		return -1;
	}

	glfwMakeContextCurrent(window);

	glfwSetKeyCallback(window, key_callback);
	glfwSetCursorPosCallback(window, cursor_position_callback);
	glfwSetWindowSizeCallback(window, window_size_callback);
	glfwSetMouseButtonCallback(window, mouse_button_callback);

	glfwSwapInterval(1);

	glewExperimental = GL_TRUE;
	GLenum err = glewInit();
	if (GLEW_OK != err) {
		printf("glewInit() failed: %s\n", glewGetErrorString(err));
		exit(-1);
	}

	glfwGetFramebufferSize(window, &width, &height);

	SetGlobalWorkingDirectoryName();
	pathToAssets = currentWorkingDir + "/..";

	try {
		exgl = new ExampleXGL();
		exgl->Reshape(width, height);
	}
	catch (std::runtime_error e) {
		printf("Exception: %s\n", e.what());
	}

	try {
		while (!glfwWindowShouldClose(window)) {
			glfwPollEvents();

			glfwSwapBuffers(window);

			exgl->Display();
		}
	}
	catch (std::runtime_error e) {
		printf("Exception: %s\n", e.what());
	}

	try {
		delete exgl;
		glfwTerminate();
	}
	catch (std::runtime_error e) {
		printf("Exception: %s\n", e.what());
	}

	return 0;
}
