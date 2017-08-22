#include "xglhmd.h"

XGLHmd::XGLHmd(XGL *p) :
	pXgl(p),
	frameIndex(0)
{
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
}

void XGLHmd::TransposeHand(ovrHandType which) {
	char* handNames[2] = { "LeftHand0", "RightHand0" };

	Matrix4f handTranslation = Matrix4f::Translation(handPoses[which].Position);
	Matrix4f ht = Matrix4f::RotationX(pi / 2) * handTranslation;
	glm::mat4 glmHandTranslation = glm::transpose(glm::make_mat4(&ht.M[0][0]));

	XGLShape* hand = (XGLShape *)pXgl->FindObject(handNames[which]);
	hand->model = glmHandTranslation;
}

void XGLHmd::TrackInput() {
	displayMidpointSeconds = ovr_GetPredictedDisplayTime(session, frameIndex);
	trackState = ovr_GetTrackingState(session, displayMidpointSeconds, ovrTrue);

	// Grab hand poses useful for rendering hand or controller representation
	handPoses[ovrHand_Left] = trackState.HandPoses[ovrHand_Left].ThePose;
	handPoses[ovrHand_Right] = trackState.HandPoses[ovrHand_Right].ThePose;

	TransposeHand(ovrHand_Left);
	TransposeHand(ovrHand_Right);

	if (OVR_SUCCESS(ovr_GetInputState(session, ovrControllerType_Touch, &inputState))) {
		if (inputState.Buttons & ovrButton_A) {
			// Handle A button being pressed
		}
		if (inputState.HandTrigger[ovrHand_Left] > 0.5f) {
			// Handle hand grip...
		}
	}
}

bool XGLHmd::Loop() {
	ovr_GetSessionStatus(session, &sessionStatus);

	if (sessionStatus.ShouldQuit)
		return true;

	if (sessionStatus.ShouldRecenter)
		ovr_RecenterTrackingOrigin(session);

	TrackInput();

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

		static OVR::Vector3f Pos2(0.0f, 0.0f, 0.0f);

		// Render Scene to Eye Buffers
		for (int eye = 0; eye < 2; ++eye) {
			// Switch to eye render target
			eyeRenderTexture[eye]->SetAndClearRenderSurface(eyeDepthBuffer[eye]);

			// Get view and projection matrices
			Matrix4f rollPitchYaw = Matrix4f::RotationZ(pi);
			Matrix4f finalRollPitchYaw = rollPitchYaw * Matrix4f(EyeRenderPose[eye].Orientation);
			Vector3f finalUp = finalRollPitchYaw.Transform(Vector3f(0, 1, 0));
			Vector3f finalForward = finalRollPitchYaw.Transform(Vector3f(0, 0, -1));
			Vector3f shiftedEyePos = Pos2 + rollPitchYaw.Transform(EyeRenderPose[eye].Position);

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

		// exit the rendering loop if submit returns an error, will retry on ovrError_DisplayLost
		if (!OVR_SUCCESS(ovr_SubmitFrame(session, frameIndex, nullptr, &layers, 1)))
			return true;

		frameIndex++;
	}

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
