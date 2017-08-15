// physx-main.cpp : Defines the entry point for the GLFW-based PhysX application.
// The console window is immediately closed.
#include <PxPhysicsAPI.h>
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

#include <physx-xgl.h>

static PhysXXGL *pxgl = NULL;
#ifndef OPENGL_MAJOR_VERSION
#define OPENGL_MAJOR_VERSION 3
#endif

#ifndef OPENGL_MINOR_VERSION
#define OPENGL_MINOR_VERSION 2
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

	window = glfwCreateWindow(1920, 1080, "Mad Style TV Example", NULL, NULL);
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
    pathToAssets = currentWorkingDir + "/..";


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
