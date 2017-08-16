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
#include <OVR_CAPI_GL.h>
#include "GLAppUtil.h"

class XGLHmd {
public:
	XGLHmd(XGL *);
	~XGLHmd() {};

	bool Loop();

private:
	void TrackInput();

	static ovrGraphicsLuid GetDefaultAdapterLuid();
	static int Compare(const ovrGraphicsLuid&, const ovrGraphicsLuid&);

	TextureBuffer* eyeRenderTexture[2];
	DepthBuffer* eyeDepthBuffer[2];
	ovrSession session;
	ovrGraphicsLuid luid;
	ovrHmdDesc hmdDesc;
	ovrSessionStatus sessionStatus;
	long long frameIndex;

	XGL *pXgl;
};

#endif