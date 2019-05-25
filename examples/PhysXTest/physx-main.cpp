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

static void mouse_button_callback(GLFWwindow *window, int button, int action, int mods) {
	int state = 0;
	double x, y;

	glfwGetCursorPos(window, &x, &y);
	state |= glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT);
	state |= glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_RIGHT) << 1;

	if (pxgl != NULL)
		pxgl->MouseEvent((int)x, (int)y, state);
}

static void window_size_callback(GLFWwindow *window, int width, int height) {
	if (pxgl != NULL) {
		pxgl->Reshape(width, height);
		pxgl->Display();
		glfwSwapBuffers(window);
	}
}

static void window_refresh_callback(GLFWwindow *window){
	if (pxgl != NULL)
		pxgl->Display();
}

static void enumerate_joysticks() {
	int i;

	for (i = 0; i < GLFW_JOYSTICK_LAST; i++) {
		if (glfwJoystickPresent(i)) {
			const char *name = glfwGetJoystickName(i);
			XJoystick j;

			strcpy(j.fullName, name);
			char *tmpPtr = j.shortName;
			
			// strip whitespace from name (probably not really needed)
			for (int j = 0; j < strlen(name); j++)
				if (name[j] != ' ')
					*tmpPtr++ = name[j];
			*tmpPtr = 0;

			// get the number of axes this joystick supports.
			j.numAxes = 0;
			j.pollFunc = [i](int* count) { 
				return glfwGetJoystickAxes(i, count); 
			};

			glfwGetJoystickAxes(i, &j.numAxes);
			pxgl->AddJoystick(j);
		}
	}
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
		pxgl = new PhysXXGL(window);
		pxgl->GetPreferredWindowSize(&width, &height);
		glfwSetWindowSize(window, width, height);
		glfwSwapInterval(pxgl->GetPreferredSwapInterval());
		pxgl->Reshape(width, height);
	}
	catch (std::runtime_error e) {
		printf("Exception: %s\n", e.what());
			}

	try {
		bool shouldQuit = false;
		while (!glfwWindowShouldClose(window) && !shouldQuit) {
			glfwPollEvents();
			pxgl->PollJoysticks();
			pxgl->Animate();

			glfwSwapBuffers(window);

			shouldQuit = pxgl->Display();
			pxgl->stepPhysics(true);
		}
	}
	catch (std::runtime_error e) {
		printf("Exception: %s\n", e.what());
		}

	try {
		delete pxgl;
		glfwTerminate();
	}
	catch (std::runtime_error e) {
		printf("Exception: %s\n", e.what());
	}

	return 0;
}
