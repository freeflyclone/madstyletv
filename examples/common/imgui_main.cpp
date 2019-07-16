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
GLFWwindow *gMainWindow{ nullptr };

#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"
const char* glsl_version = "#version 150";
extern void ShowExampleMenuFile();
void ShowMainMenuBar();
ImGuiIO* pIo{ nullptr };

#include <xgl.h>
#include <ExampleXGL.h>

static ExampleXGL *exgl = NULL;
#ifndef OPENGL_MAJOR_VERSION
#define OPENGL_MAJOR_VERSION 3
#endif

#ifndef OPENGL_MINOR_VERSION
#define OPENGL_MINOR_VERSION 2
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
	fprintf(stderr, "GLFW_ErrorCallback: %s\n", description);
}

static void key_callback(GLFWwindow *window, int key, int scancode, int action, int mods) {
	if (pIo->WantCaptureKeyboard)
		return;

	if (exgl != NULL) {
		if (action == GLFW_PRESS)
			exgl->KeyEvent(key, 0);
		else if (action == GLFW_RELEASE)
			exgl->KeyEvent(key, 0x8000);
	}
}

static void cursor_position_callback(GLFWwindow *window, double x, double y) {
	if (pIo->WantCaptureMouse)
		return;

	int state = 0;

	state |= glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT);
	state |= glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_RIGHT) << 1;

	if (exgl != NULL)
		exgl->MouseEvent((int)x, (int)y, state);
}

static void mouse_button_callback(GLFWwindow *window, int button, int action, int mods) {
	if (pIo->WantCaptureMouse)
		return;

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
			exgl->AddJoystick(j);
		}
	}
}

typedef std::map<GLenum, std::string> DebugSourceStrings;
DebugSourceStrings debugSourceStrings = {
	{ GL_DEBUG_SOURCE_API, "API" },
	{ GL_DEBUG_SOURCE_WINDOW_SYSTEM, "Window System" },
	{ GL_DEBUG_SOURCE_SHADER_COMPILER, "Shader Compiler" },
	{ GL_DEBUG_SOURCE_THIRD_PARTY, "Third Party" },
	{ GL_DEBUG_SOURCE_APPLICATION, "Application" },
	{ GL_DEBUG_SOURCE_OTHER, "Other" },
};
typedef std::map<GLenum, std::string> DebugTypeStrings;
DebugSourceStrings debugTypeStrings = {
	{ GL_DEBUG_TYPE_ERROR, "Error" },
	{ GL_DEBUG_TYPE_DEPRECATED_BEHAVIOR, "Deprecated Behavior" },
	{ GL_DEBUG_TYPE_UNDEFINED_BEHAVIOR, "Undefined Behavior" },
	{ GL_DEBUG_TYPE_PORTABILITY, "Portability" },
	{ GL_DEBUG_TYPE_PERFORMANCE, "Performance" },
	{ GL_DEBUG_TYPE_MARKER, "Marker" },
	{ GL_DEBUG_TYPE_PUSH_GROUP, "Push Group" },
	{ GL_DEBUG_TYPE_POP_GROUP, "Pop Group" },
	{ GL_DEBUG_TYPE_OTHER, "Other" },
};
typedef std::map<GLenum, std::string> DebugSeverityStrings;
DebugSourceStrings debugSeverityStrings = {
	{ GL_DEBUG_SEVERITY_HIGH, "High" },
	{ GL_DEBUG_SEVERITY_MEDIUM, "Medium" },
	{ GL_DEBUG_SEVERITY_LOW, "Low" },
	{ GL_DEBUG_SEVERITY_NOTIFICATION, "Notification" },
};

void GLAPIENTRY GLDebugCallback(GLenum source, GLenum type, GLuint id, GLenum severity, GLsizei length, const GLchar* message, const void* userParam) {
	GLFWwindow *window = (GLFWwindow*)userParam;
	xprintf("%s(): %s:%s:%s\n\t%s\n",
		__FUNCTION__,
		debugSourceStrings[source].c_str(),
		debugTypeStrings[type].c_str(),
		debugSeverityStrings[severity].c_str(),
		message);
}

void InitGLDebugLog(GLFWwindow *window) {
#ifdef OPENGL_DEBUG_LOG
	if (GLEW_ARB_debug_output) {
		xprintf("GLEW_ARB_debug_output available\n");

		glDebugMessageCallback(GLDebugCallback, window);

		glEnable(GL_DEBUG_OUTPUT);
		if (glIsEnabled(GL_DEBUG_OUTPUT))
			xprintf("GL_DEBUG_OUTPUT enabled\n");

		glEnable(GL_DEBUG_OUTPUT_SYNCHRONOUS);
		if (glIsEnabled(GL_DEBUG_OUTPUT_SYNCHRONOUS))
			xprintf("GL_DEBUG_OUTPUT_SYNCHRONOUS enabled\n");

		glDebugMessageControl(GL_DONT_CARE, GL_DONT_CARE, GL_DONT_CARE, 0, nullptr, GL_TRUE);
	}
#endif
}

void GLLogString(const char* msg) {
	glDebugMessageInsert(GL_DEBUG_SOURCE_APPLICATION, GL_DEBUG_TYPE_OTHER, 0, GL_DEBUG_SEVERITY_NOTIFICATION, 128, msg);
}

int main(void) {
	int width, height;

	if (!FreeConsole()) {
		printf("Freeing the console failed: %d\n", GetLastError());
		exit(0);
	}

	if (!glfwInit()) {
		printf("glfwInit() failed\n");
		return -1;
	}

	glfwSetErrorCallback(error_callback);

	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, OPENGL_MAJOR_VERSION);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, OPENGL_MINOR_VERSION);
	glfwWindowHint(GLFW_SAMPLES, 8);
#ifdef OPENGL_DEBUG_LOG
	glfwWindowHint(GLFW_OPENGL_DEBUG_CONTEXT, GLFW_TRUE);
#endif

	IMGUI_CHECKVERSION();
	ImGui::CreateContext();


	gMainWindow = glfwCreateWindow(1280, 720, GLFW_WINDOW_TITLE, NULL, NULL);
	if (!gMainWindow) {
		printf("glfwCreateWindow() failed\n");
		glfwTerminate();
		return -1;
	}
	glfwSetWindowPos(gMainWindow, 16, 64);
	glfwMakeContextCurrent(gMainWindow);

	glfwSetKeyCallback(gMainWindow, key_callback);
	glfwSetCursorPosCallback(gMainWindow, cursor_position_callback);
	glfwSetWindowSizeCallback(gMainWindow, window_size_callback);
	glfwSetMouseButtonCallback(gMainWindow, mouse_button_callback);

	glewExperimental = GL_TRUE;
	GLenum err = glewInit();
	if (GLEW_OK != err) {
		printf("glewInit() failed: %s\n", glewGetErrorString(err));
		exit(-1);
	}

	InitGLDebugLog(gMainWindow);

	glfwGetFramebufferSize(gMainWindow, &width, &height);

	SetGlobalWorkingDirectoryName();
	pathToAssets = currentWorkingDir + "/..";

	pIo = &ImGui::GetIO();
	ImGui::StyleColorsDark();

	ImGui_ImplGlfw_InitForOpenGL(gMainWindow, true);
	ImGui_ImplOpenGL3_Init(glsl_version);

	ImFont* font = pIo->Fonts->AddFontFromFileTTF("c:\\Windows\\Fonts\\Arial.ttf", 18.0f);
	IM_ASSERT(font != NULL);

	try {
		GLLogString("Initializing XGL...");
		exgl = new ExampleXGL(gMainWindow);
		enumerate_joysticks();
		exgl->GetPreferredWindowSize(&width, &height);
		glfwSetWindowSize(gMainWindow, width, height);
		glfwSwapInterval(exgl->GetPreferredSwapInterval());
		exgl->Reshape(width, height);

		bool shouldQuit = false;

		GLLogString("...XGL init complete, loop starting.");

		while (!glfwWindowShouldClose(gMainWindow) && !shouldQuit) {
			ImGui_ImplOpenGL3_NewFrame();
			ImGui_ImplGlfw_NewFrame();
			ImGui::NewFrame();

			glfwPollEvents();
			exgl->PollJoysticks();
			exgl->Animate();

			shouldQuit = exgl->Display();

			ShowMainMenuBar();

			ImGui::Render();
			ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

			glfwSwapBuffers(gMainWindow);
		}
	}
	catch (std::runtime_error e) {
		printf("Exception: %s\n", e.what());
		xprintf("Exception: %s\n", e.what());
	}

	try {
		delete exgl;
		glfwTerminate();
	}
	catch (std::runtime_error e) {
		printf("Exception: %s\n", e.what());
		xprintf("Exception: %s\n", e.what());
	}

	return 0;
}
