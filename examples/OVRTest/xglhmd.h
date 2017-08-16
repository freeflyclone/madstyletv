#ifndef XGLHMD_H
#define XGLHMD_H

#include "xgl.h"

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

class XGLHmd {
public:
	XGLHmd();
	~XGLHmd() {};

	bool Loop(XGL*);

private:
	bool shouldQuit = false;
	TextureBuffer* eyeRenderTexture[2];
	DepthBuffer* eyeDepthBuffer[2];
	ovrResult result;
	ovrSession session;
	ovrGraphicsLuid luid;
	ovrHmdDesc hmdDesc;
	ovrSessionStatus sessionStatus;
	long long frameIndex = 0;
};

#endif