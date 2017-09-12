#include "xglhmd.h"

XGLHmd::XGLHmd(XGL *p, int w, int h) :
	pXgl(p),
	frameIndex(0),
	width(w),
	height(h)
{
	// the hmdSled is an XGLShape that the HMD and Touch controllers are
	// attached to.  XGLShape objects representing the Touch controllers
	// are assumed to be attached as child XObjects, thus when the sled
	// moves, the "hands" move with it.
	hmdSled = (XGLShape *)pXgl->FindObject("HmdSled0");

	handNames[0] = "LeftHand0";
	handNames[1] = "RightHand0";

	whichHand[0] = "Left";
	whichHand[1] = "Right";

	memset(&previousState, 0, sizeof(previousState));

	if (!OVR_SUCCESS(ovr_Initialize(nullptr)))
		throw std::runtime_error("Failed to initialize libOVR");

	if (!OVR_SUCCESS(ovr_Create(&session, &luid)))
		throw std::runtime_error("Failed to create OVR Session");

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
			throw std::runtime_error("eyeRenderTexture creation failed.");
	}

	// FloorLevel will give tracking poses where the floor height is 0
	ovr_SetTrackingOriginType(session, ovrTrackingOrigin_FloorLevel);

	memset(&desc, 0, sizeof(desc));
	desc.Width = width;
	desc.Height = height;
	desc.Format = OVR_FORMAT_R8G8B8A8_UNORM_SRGB;

	// Create mirror texture and an FBO used to copy mirror texture to back buffer
	if (!OVR_SUCCESS(ovr_CreateMirrorTextureGL(session, &desc, &mirrorTexture)))
		throw std::runtime_error("Failed to create mirror texture.");

	// Configure the mirror read buffer
	GLuint texId;
	ovr_GetMirrorTextureBufferGL(session, mirrorTexture, &texId);

	glGenFramebuffers(1, &mirrorFBO);
	glBindFramebuffer(GL_READ_FRAMEBUFFER, mirrorFBO);
	glFramebufferTexture2D(GL_READ_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, texId, 0);
	glFramebufferRenderbuffer(GL_READ_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, 0);
	glBindFramebuffer(GL_READ_FRAMEBUFFER, 0);
}

void XGLHmd::TrackTouchTriggers(ovrHandType which) {
	if (inputState.HandTrigger[which] > 0.0011f) {
		pXgl->ProportionalEvent(whichHand[which] + "HandTrigger", inputState.HandTrigger[which]);
		previousState.HandTrigger[which] = inputState.HandTrigger[which];
	}
	if (inputState.IndexTrigger[which] > 0.0011f) {
		pXgl->ProportionalEvent(whichHand[which] + "IndexTrigger", inputState.IndexTrigger[which]);
		previousState.IndexTrigger[which] = inputState.IndexTrigger[which];
	}
}

void XGLHmd::TrackTouchThumbStick(ovrHandType which) {
	if (fabs(inputState.Thumbstick[which].x) > 0.0011f) {
		pXgl->ProportionalEvent(whichHand[which] + "ThumbStick.x", inputState.Thumbstick[which].x);
		previousState.Thumbstick[which].x = inputState.Thumbstick[which].x;
	}
	if (fabs(inputState.Thumbstick[which].y) > 0.0011f) {
		pXgl->ProportionalEvent(whichHand[which] + "ThumbStick.y", inputState.Thumbstick[which].y);
		previousState.Thumbstick[which].y = inputState.Thumbstick[which].y;
	}
}

void XGLHmd::TransposeHand(ovrHandType which) {
	// read hand orientation
	ovrQuatf oq = handPoses[which].Orientation;

	// get current hand position (pose info)
	Vector3f handPos = handPoses[which].Position;

	// apply rotation matrix to covert to XGL world coordinate scheme.
	Matrix4f ht = Matrix4f::RotationX(pi / 2) * Matrix4f::Translation(handPos);

	// Fetch the current XGLShape for the hand in question by name
	XGLShape* hand = (XGLShape *)pXgl->FindObject(handNames[which]);

	// transform hand by translation * orientation
	hand->model = glm::transpose(glm::make_mat4(&ht.M[0][0])) * glm::toMat4(glm::quat(oq.w, oq.x, oq.y, oq.z));
}

void XGLHmd::TrackTouchInput() {
	displayMidpointSeconds = ovr_GetPredictedDisplayTime(session, frameIndex);
	trackState = ovr_GetTrackingState(session, displayMidpointSeconds, ovrTrue);

	// Grab hand poses useful for rendering hand or controller representation
	handPoses[ovrHand_Left] = trackState.HandPoses[ovrHand_Left].ThePose;
	handPoses[ovrHand_Right] = trackState.HandPoses[ovrHand_Right].ThePose;

	if (OVR_SUCCESS(ovr_GetInputState(session, ovrControllerType_Touch, &inputState))) {
		if (inputState.Buttons & ovrButton_A) {
			// Handle A button being pressed
		}
		TrackTouchTriggers(ovrHand_Left);
		TrackTouchTriggers(ovrHand_Right);
		TrackTouchThumbStick(ovrHand_Left);
		TrackTouchThumbStick(ovrHand_Right);
	}

	TransposeHand(ovrHand_Left);
	TransposeHand(ovrHand_Right);
}

bool XGLHmd::Loop() {
	ovr_GetSessionStatus(session, &sessionStatus);

	if (sessionStatus.ShouldQuit)
		return true;

	if (sessionStatus.ShouldRecenter)
		ovr_RecenterTrackingOrigin(session);

	if (sessionStatus.IsVisible) {
		// Call ovr_GetRenderDesc each frame to get the ovrEyeRenderDesc, as the returned values (e.g. HmdToEyePose) may change at runtime.
		ovrEyeRenderDesc eyeRenderDesc[2];
		eyeRenderDesc[0] = ovr_GetRenderDesc(session, ovrEye_Left, hmdDesc.DefaultEyeFov[0]);
		eyeRenderDesc[1] = ovr_GetRenderDesc(session, ovrEye_Right, hmdDesc.DefaultEyeFov[1]);

		// Get eye poses, feeding in correct IPD offset
		ovrPosef EyeRenderPose[2];
		ovrPosef HmdToEyePose[2] = { eyeRenderDesc[0].HmdToEyePose, eyeRenderDesc[1].HmdToEyePose };

		double sensorSampleTime;    // sensorSampleTime is fed into the layer later
		ovr_GetEyePoses(session, frameIndex, ovrTrue, HmdToEyePose, EyeRenderPose, &sensorSampleTime);

		// get head position from hmdSled
		Vector3f headPosition = { -hmdSled->model[3][0], -hmdSled->model[3][2], -hmdSled->model[3][1] };

		// Render Scene to Eye Buffers
		for (int eye = 0; eye < 2; ++eye) {
			// Switch to eye render target
			eyeRenderTexture[eye]->SetAndClearRenderSurface(eyeDepthBuffer[eye]);

			// Get view and projection matrices
			Matrix4f rollPitchYaw = Matrix4f::RotationZ(pi);
			Matrix4f finalRollPitchYaw = rollPitchYaw * Matrix4f(EyeRenderPose[eye].Orientation);
			Vector3f finalUp = finalRollPitchYaw.Transform(Vector3f(0, 1, 0));
			Vector3f finalForward = finalRollPitchYaw.Transform(Vector3f(0, 0, -1));
			Vector3f shiftedEyePos = headPosition + rollPitchYaw.Transform(EyeRenderPose[eye].Position);

			Matrix4f view = Matrix4f::LookAtRH(shiftedEyePos, shiftedEyePos + finalForward, finalUp);
			Matrix4f proj = ovrMatrix4f_Projection(hmdDesc.DefaultEyeFov[eye], 0.2f, 1000.0f, ovrProjection_None);

			// build XGL view and projection matrix...
			// "myView" converts to XGL world coordinates, where the ground plane is X,Y and "up" is the Z axis
			//    from customary OpenGL RH coordinate system where X,Z are the ground plane and Y is up
			Matrix4f myView = view * Matrix4f::RotationX(pi / 2) * Matrix4f::RotationZ(pi);

			// set the projection,view,orthoProjection matrices in the matrix UBO
			pXgl->shaderMatrix.view = glm::transpose(glm::make_mat4(&myView.M[0][0]));;
			pXgl->shaderMatrix.projection = glm::transpose(glm::make_mat4(&proj.M[0][0]));
			pXgl->shaderMatrix.orthoProjection = pXgl->projector.GetOrthoMatrix();

			// render XGL scene
			pXgl->DisplayOVR();

			eyeRenderTexture[eye]->UnsetRenderSurface();

			// Commit changes to the textures so they get picked up frame
			eyeRenderTexture[eye]->Commit();
		}
		ovrLayerEyeFov ld;
		ld.Header.Type = ovrLayerType_EyeFov;
		ld.Header.Flags = ovrLayerFlag_TextureOriginAtBottomLeft;   // Because OpenGL.

		for (int eye = 0; eye < 2; ++eye) {
			ld.ColorTexture[eye] = eyeRenderTexture[eye]->TextureChain;
			ld.Viewport[eye] = OVR::Recti(eyeRenderTexture[eye]->GetSize());
			ld.Fov[eye] = hmdDesc.DefaultEyeFov[eye];
			ld.RenderPose[eye] = EyeRenderPose[eye];
			ld.SensorSampleTime = sensorSampleTime;
		}

		ovrLayerHeader* layers = &ld.Header;

		TrackTouchInput();

		// exit the rendering loop if submit returns an error, will retry on ovrError_DisplayLost
		if (!OVR_SUCCESS(ovr_SubmitFrame(session, frameIndex, nullptr, &layers, 1)))
			return true;

		frameIndex++;
	}

	// Blit mirror texture to back buffer
	glBindFramebuffer(GL_READ_FRAMEBUFFER, mirrorFBO);
	glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0);
	GLint w = width;
	GLint h = height;
	glBlitFramebuffer(0, h, w, 0,
		0, 0, w, h,
		GL_COLOR_BUFFER_BIT, GL_NEAREST);
	glBindFramebuffer(GL_READ_FRAMEBUFFER, 0);


	return false;
}

ovrGraphicsLuid XGLHmd::GetDefaultAdapterLuid() {
	ovrGraphicsLuid luid = ovrGraphicsLuid();

#if defined(_WIN32)
	IDXGIFactory* factory = nullptr;

	if (SUCCEEDED(CreateDXGIFactory(IID_PPV_ARGS(&factory)))) {
		IDXGIAdapter* adapter = nullptr;

		if (SUCCEEDED(factory->EnumAdapters(0, &adapter))) {
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

int XGLHmd::Compare(const ovrGraphicsLuid& lhs, const ovrGraphicsLuid& rhs) {
	return memcmp(&lhs, &rhs, sizeof(ovrGraphicsLuid));
}
