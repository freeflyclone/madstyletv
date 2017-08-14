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

#if defined(_WIN32)
#include <dxgi.h> // for GetDefaultAdapterLuid
#pragma comment(lib, "dxgi.lib")
#endif

#include <OVR_CAPI.h>
#include "OVR_Version.h"
#include "OVR_ErrorCode.h"
//#include "OVR_CAPI_Prototypes.h"
#include <OVR_CAPI_GL.h>
#include "GLAppUtil.h"

static ovrGraphicsLuid GetDefaultAdapterLuid()
{
	ovrGraphicsLuid luid = ovrGraphicsLuid();

#if defined(_WIN32)
	IDXGIFactory* factory = nullptr;

	if (SUCCEEDED(CreateDXGIFactory(IID_PPV_ARGS(&factory))))
	{
		IDXGIAdapter* adapter = nullptr;

		if (SUCCEEDED(factory->EnumAdapters(0, &adapter)))
		{
			DXGI_ADAPTER_DESC desc;

			adapter->GetDesc(&desc);
			memcpy(&luid, &desc.AdapterLuid, sizeof(luid));
			adapter->Release();
		}

		factory->Release();
	}
#endif

	return luid;
}


static int Compare(const ovrGraphicsLuid& lhs, const ovrGraphicsLuid& rhs)
{
	return memcmp(&lhs, &rhs, sizeof(ovrGraphicsLuid));
}

bool shouldQuit = false;
TextureBuffer * eyeRenderTexture[2] = { nullptr, nullptr };
DepthBuffer   * eyeDepthBuffer[2] = { nullptr, nullptr };
ovrResult result;
ovrSession session;
ovrGraphicsLuid luid;
ovrHmdDesc hmdDesc;
ovrSessionStatus sessionStatus;
long long frameIndex = 0;

bool InitOvr(bool retryCreate) {
	result = ovr_Initialize(nullptr);
	if (!OVR_SUCCESS(result))
		throw std::runtime_error("Failed to initialize libOVR");

	result = ovr_Create(&session, &luid);
	if (!OVR_SUCCESS(result))
		return retryCreate;

	if (Compare(luid, GetDefaultAdapterLuid())) // If luid that the Rift is on is not the default adapter LUID...
		throw std::runtime_error("OpenGL supports only the default graphics adapter.");

	hmdDesc = ovr_GetHmdDesc(session);

	// Make eye render buffers
	for (int eye = 0; eye < 2; ++eye)
	{
		ovrSizei idealTextureSize = ovr_GetFovTextureSize(session, ovrEyeType(eye), hmdDesc.DefaultEyeFov[eye], 1);
		eyeRenderTexture[eye] = new TextureBuffer(session, true, true, idealTextureSize, 1, NULL, 1);
		eyeDepthBuffer[eye] = new DepthBuffer(eyeRenderTexture[eye]->GetSize(), 0);

		if (!eyeRenderTexture[eye]->TextureChain)
			throw std::runtime_error("TextureChain creation failed.");
	}

	// FloorLevel will give tracking poses where the floor height is 0
	ovr_SetTrackingOriginType(session, ovrTrackingOrigin_FloorLevel);
}

void OvrLoop() {
	static float Yaw(3.141592f);

	ovr_GetSessionStatus(session, &sessionStatus);

	if (sessionStatus.ShouldQuit) {
		shouldQuit = true;
		return;
	}
	if (sessionStatus.ShouldRecenter)
		ovr_RecenterTrackingOrigin(session);

	if (sessionStatus.IsVisible) {
		// Call ovr_GetRenderDesc each frame to get the ovrEyeRenderDesc, as the returned values (e.g. HmdToEyePose) may change at runtime.
		ovrEyeRenderDesc eyeRenderDesc[2];
		eyeRenderDesc[0] = ovr_GetRenderDesc(session, ovrEye_Left, hmdDesc.DefaultEyeFov[0]);
		eyeRenderDesc[1] = ovr_GetRenderDesc(session, ovrEye_Right, hmdDesc.DefaultEyeFov[1]);

		// Get eye poses, feeding in correct IPD offset
		ovrPosef EyeRenderPose[2];
		ovrPosef HmdToEyePose[2] = { eyeRenderDesc[0].HmdToEyePose,
			eyeRenderDesc[1].HmdToEyePose };

		double sensorSampleTime;    // sensorSampleTime is fed into the layer later
		ovr_GetEyePoses(session, frameIndex, ovrTrue, HmdToEyePose, EyeRenderPose, &sensorSampleTime);

		static OVR::Vector3f Pos2(0.0f, 0.0f, -5.0f);

		// Render Scene to Eye Buffers
		for (int eye = 0; eye < 2; ++eye)
		{
			// Switch to eye render target
			eyeRenderTexture[eye]->SetAndClearRenderSurface(eyeDepthBuffer[eye]);

			// Get view and projection matrices
			OVR::Matrix4f rollPitchYaw = OVR::Matrix4f::RotationY(Yaw);
			OVR::Matrix4f finalRollPitchYaw = rollPitchYaw * OVR::Matrix4f(EyeRenderPose[eye].Orientation);
			OVR::Vector3f finalUp = finalRollPitchYaw.Transform(OVR::Vector3f(0, 1, 0));
			OVR::Vector3f finalForward = finalRollPitchYaw.Transform(OVR::Vector3f(0, 0, -1));
			OVR::Vector3f shiftedEyePos = Pos2 + rollPitchYaw.Transform(EyeRenderPose[eye].Position);

			OVR::Matrix4f view = OVR::Matrix4f::LookAtRH(shiftedEyePos, shiftedEyePos + finalForward, finalUp);
			OVR::Matrix4f proj = ovrMatrix4f_Projection(hmdDesc.DefaultEyeFov[eye], 0.2f, 1000.0f, ovrProjection_None);

			exgl->Display();

			eyeRenderTexture[eye]->UnsetRenderSurface();

			// Commit changes to the textures so they get picked up frame
			eyeRenderTexture[eye]->Commit();
		}
		ovrLayerEyeFov ld;
		ld.Header.Type = ovrLayerType_EyeFov;
		ld.Header.Flags = ovrLayerFlag_TextureOriginAtBottomLeft;   // Because OpenGL.

		for (int eye = 0; eye < 2; ++eye)
		{
			ld.ColorTexture[eye] = eyeRenderTexture[eye]->TextureChain;
			ld.Viewport[eye] = OVR::Recti(eyeRenderTexture[eye]->GetSize());
			ld.Fov[eye] = hmdDesc.DefaultEyeFov[eye];
			ld.RenderPose[eye] = EyeRenderPose[eye];
			ld.SensorSampleTime = sensorSampleTime;
		}

		ovrLayerHeader* layers = &ld.Header;
		result = ovr_SubmitFrame(session, frameIndex, nullptr, &layers, 1);
		// exit the rendering loop if submit returns an error, will retry on ovrError_DisplayLost

		if (!OVR_SUCCESS(result))
			shouldQuit = true;

		frameIndex++;
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

		InitOvr(false);
	}
	catch (std::runtime_error e) {
		printf("Exception: %s\n", e.what());
	}

	try {
		while (!glfwWindowShouldClose(window)) {
			glfwPollEvents();

			glfwSwapBuffers(window);

			OvrLoop();
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
