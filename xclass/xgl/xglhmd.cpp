#include "xgl.h"
#include "xglhmd.h"

XGLHmd::XGLHmd(XGL *p, int w, int h) :
	pXgl(p),
	frameIndex(0),
	width(w),
	height(h)
{
	XGLShape *shape, *leftFinger, *leftFingerTip, *leftThumb, *rightFinger, *rightFingerTip, *rightThumb;

	handNames[0] = "LeftHand0";
	handNames[1] = "RightHand0";

	whichHand[0] = "Left";
	whichHand[1] = "Right";

	// LeftHand, anchored to the cockpit
	pXgl->CreateShape("shaders/specular", [&]() { shape = new XGLSphere(0.05f, 16); shape->SetName("LeftHand"); return shape; });
	pXgl->hmdSled->AddChild(shape);

	// Left Finger, child of LeftHand
	pXgl->CreateShape("shaders/specular", [&]() { leftFinger = new XGLCapsule(0.01f, 0.1f, 8); leftFinger->SetName("LeftFinger"); return leftFinger; });
	glm::mat4 translate = glm::translate(glm::mat4(), glm::vec3(0.0, 0.098, 0.0));
	glm::mat4 rotate = glm::rotate(glm::mat4(), glm::radians(90.0f), glm::vec3(0.0, 0.0, 1.0));
	leftFinger->model = translate * rotate;
	shape->AddChild(leftFinger);

	// RightFinterTip, child of RightFinger
	pXgl->CreateShape("shaders/specular", [&]() { leftFingerTip = new XGLSphere(0.012f, 16); leftFingerTip->SetName("LeftFingerTip"); return leftFingerTip; });
	translate = glm::translate(glm::mat4(), glm::vec3(0.05, 0.0, 0.0));
	leftFingerTip->model = translate;
	leftFinger->AddChild(leftFingerTip);

	// LeftThumb, child of LeftHand
	pXgl->CreateShape("shaders/specular", [&]() { leftThumb = new XGLCapsule(0.01f, 0.075f, 8); leftThumb->SetName("LeftThumb"); return leftThumb; });
	translate = glm::translate(glm::mat4(), glm::vec3(0.06, 0.06, 0.0));
	rotate = glm::rotate(glm::mat4(), glm::radians(45.0f), glm::vec3(0.0, 0.0, 1.0));
	leftThumb->model = translate * rotate;
	shape->AddChild(leftThumb);


	// RightHand, anchored to the cockpit
	pXgl->CreateShape("shaders/specular", [&]() { shape = new XGLSphere(0.05f, 16); shape->SetName("RightHand"); return shape; });
	pXgl->hmdSled->AddChild(shape);

	// RightFinger, child of RightHand
	pXgl->CreateShape("shaders/specular", [&]() { rightFinger = new XGLCapsule(0.01f, 0.1f, 8); rightFinger->SetName("RightFinger"); return rightFinger; });
	translate = glm::translate(glm::mat4(), glm::vec3(0.0, 0.098, 0.0));
	rotate = glm::rotate(glm::mat4(), glm::radians(90.0f), glm::vec3(0.0, 0.0, 1.0));
	rightFinger->model = translate * rotate;
	shape->AddChild(rightFinger);

	// RightFinterTip, child of RightFinger
	pXgl->CreateShape("shaders/specular", [&]() { rightFingerTip = new XGLSphere(0.012f, 16); rightFingerTip->SetName("RightFingerTip"); return rightFingerTip; });
	translate = glm::translate(glm::mat4(), glm::vec3(0.05, 0.0, 0.0));
	rightFingerTip->model = translate;
	rightFinger->AddChild(rightFingerTip);

	// RightThumb, child of Right Hand
	pXgl->CreateShape("shaders/specular", [&]() { rightThumb = new XGLCapsule(0.01f, 0.075f, 8); rightThumb->SetName("RightThumb"); return rightThumb; });
	translate = glm::translate(glm::mat4(), glm::vec3(-0.06, 0.06, 0.0));
	rotate = glm::rotate(glm::mat4(), glm::radians(135.0f), glm::vec3(0.0, 0.0, 1.0));
	rightThumb->model = translate * rotate;
	shape->AddChild(rightThumb);

	// attach HUD to sled
	//hmdSled->AddChild(appGuiManager);

	// Fetch the current XGLShapes for the hands
	hands[0] = (XGLShape *)pXgl->FindObject(handNames[0]);
	hands[1] = (XGLShape *)pXgl->FindObject(handNames[1]);

	hmdSled = (XGLSled *)pXgl->FindObject("HmdSled");

	memset(&previousState, 0, sizeof(previousState));

	if (!OVR_SUCCESS(ovr_Initialize(nullptr)))
		throw std::runtime_error("Failed to initialize libOVR");

	ovrResult result;
	if ((result = ovr_Create(&session, &luid)) < 0) {
		xprintf("ovr_Create() failed: %d\n", result);
		throw std::runtime_error("Failed to create OVR Session");
	}

	if (Compare(luid, GetDefaultAdapterLuid())) // If luid that the Rift is on is not the default adapter LUID...
		throw std::runtime_error("OpenGL supports only the default graphics adapter.");

	hmdDesc = ovr_GetHmdDesc(session);

	// Make eye render buffers
	for (int eye = 0; eye < 2; ++eye)
	{
		ovrSizei idealTextureSize = ovr_GetFovTextureSize(session, ovrEyeType(eye), hmdDesc.DefaultEyeFov[eye], 1.5f);
		eyeRenderTexture[eye] = new TextureBuffer(session, true, true, idealTextureSize, 1, NULL, 1);
		eyeDepthBuffer[eye] = new DepthBuffer(eyeRenderTexture[eye]->GetSize(), 0);

		if (!eyeRenderTexture[eye]->TextureChain)
			throw std::runtime_error("eyeRenderTexture creation failed.");
	}

	// FloorLevel will give tracking poses where the floor height is 0
	ovr_SetTrackingOriginType(session, ovrTrackingOrigin_EyeLevel);

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

void XGLHmd::TrackTouchButtons() {
	static bool aState = false;
	static bool bState = false;
	static bool xState = false;
	static bool yState = false;
	static bool enterState = false;

	if (inputState.Buttons & ovrButton_A) {
		if (!aState) {
			pXgl->KeyEvent(1, 0);
			aState = true;
		}
	}
	else if (aState){
		pXgl->KeyEvent(1, 0x8000);
		aState = false;
	}

	if (inputState.Buttons & ovrButton_B) {
		if (!bState) {
			pXgl->KeyEvent(2, 0);
			bState = true;
		}
	}
	else if (bState){
		pXgl->KeyEvent(2, 0x8000);
		bState = false;
	}

	if (inputState.Buttons & ovrButton_X) {
		if (!xState) {
			pXgl->KeyEvent(3, 0);
			xState = true;
		}
	}
	else if (xState){
		pXgl->KeyEvent(3, 0x8000);
		xState = false;
	}

	if (inputState.Buttons & ovrButton_Y) {
		if (!yState) {
			pXgl->KeyEvent(4, 0);
			yState = true;
		}
	}
	else if (yState) {
		pXgl->KeyEvent(3, 0x8000);
		yState = false;
	}

	// if the left controller button is down...
	if (inputState.Buttons & ovrButton_Enter) {
		// ... and it wasn't down last time
		if (!enterState) {
			// its just been pressed, so fire a key-press event
			pXgl->KeyEvent('h', 0);
			// mark button pressed
			enterState = true;
		}
	}
	// if it's not down now but was down last time...
	else if (enterState) {
		// it's just been released, so fire a key-release event
		pXgl->KeyEvent('h', 0x8000);
		// mark button released
		enterState = false;
	}
}

void XGLHmd::TrackTouchInput() {
	displayMidpointSeconds = ovr_GetPredictedDisplayTime(session, frameIndex);
	trackState = ovr_GetTrackingState(session, displayMidpointSeconds, ovrTrue);

	// Grab hand poses useful for rendering hand or controller representation
	handPoses[ovrHand_Left] = trackState.HandPoses[ovrHand_Left].ThePose;
	handPoses[ovrHand_Right] = trackState.HandPoses[ovrHand_Right].ThePose;

	if (OVR_SUCCESS(ovr_GetInputState(session, ovrControllerType_Touch, &inputState))) {
		TrackTouchButtons();
		TrackTouchTriggers(ovrHand_Left);
		TrackTouchTriggers(ovrHand_Right);
		TrackTouchThumbStick(ovrHand_Left);
		TrackTouchThumbStick(ovrHand_Right);
	}

	TransposeHand(ovrHand_Left);
	TransposeHand(ovrHand_Right);
}

void XGLHmd::TransposeHand(ovrHandType which) {
	// read hand orientation, position from OVR
	ovrQuatf oq = handPoses[which].Orientation;
	Vector3f handPos = handPoses[which].Position;

	if (hands[which] == nullptr)
		return;

	XGLShape *hand = hands[which];

	// convert OVR orientation & position to GLM form
	hand->o = glm::quat(oq.w, oq.x, -oq.z, oq.y);
	hand->p = glm::vec3(handPos.x, -handPos.z, handPos.y);

	// transform hand by translation * orientation
	hand->model = glm::translate(glm::mat4(), hand->p) * glm::toMat4(hand->o);
}

void XGLHmd::TransformEye(int eye) {
	glm::mat4 projection = glm::transpose(glm::make_mat4(&ovrMatrix4f_Projection(hmdDesc.DefaultEyeFov[eye], 0.2f, 1000.0f, ovrProjection_None).M[0][0]));
	glm::vec3 position = { EyeRenderPose[eye].Position.x, EyeRenderPose[eye].Position.y, EyeRenderPose[eye].Position.z };
	glm::fquat orientation = { EyeRenderPose[eye].Orientation.w, EyeRenderPose[eye].Orientation.x, EyeRenderPose[eye].Orientation.y, EyeRenderPose[eye].Orientation.z };

	// This makes lighting work correctly!
	pXgl->camera.pos = hmdSled->p + position;

	// "tweakView" converts to XGL world coordinates, where the ground plane is X,Y and "up" is the Z axis
	//    from customary OpenGL RH coordinate system where X,Z are the ground plane and Y is up
	glm::mat4 tweakView = glm::rotate(glm::mat4(), -pi / 2, glm::vec3(1.0, 0.0, 0.0));

	glm::mat4 eyeO = glm::toMat4(orientation);
	glm::vec3 eyeU(eyeO * glm::vec4(0, 1, 0, 1));
	glm::vec3 eyeF(eyeO * glm::vec4(0, 0, -1, 1));
	glm::mat4 gView = glm::lookAt(position, position + eyeF, eyeU) * tweakView;

	// set the projection,view,orthoProjection matrices in the matrix UBO
	pXgl->shaderMatrix.view = gView * glm::inverse(hmdSled->model);
	pXgl->shaderMatrix.projection = projection;
	pXgl->shaderMatrix.orthoProjection = pXgl->projector.GetOrthoMatrix();
}

bool XGLHmd::Loop() {
	ovr_GetSessionStatus(session, &sessionStatus);

	if (sessionStatus.ShouldQuit)
		return true;

	if (sessionStatus.ShouldRecenter)
		ovr_RecenterTrackingOrigin(session);

	if (sessionStatus.IsVisible) {
		pXgl->Animate();

		// Call ovr_GetRenderDesc each frame to get the ovrEyeRenderDesc, as the returned values (e.g. HmdToEyePose) may change at runtime.
		ovrEyeRenderDesc eyeRenderDesc[2];
		eyeRenderDesc[0] = ovr_GetRenderDesc(session, ovrEye_Left, hmdDesc.DefaultEyeFov[0]);
		eyeRenderDesc[1] = ovr_GetRenderDesc(session, ovrEye_Right, hmdDesc.DefaultEyeFov[1]);

		// Get eye poses, feeding in correct IPD offset
		ovrPosef HmdToEyePose[2] = { eyeRenderDesc[0].HmdToEyePose, eyeRenderDesc[1].HmdToEyePose };

		double sensorSampleTime;    // sensorSampleTime is fed into the layer later
		ovr_GetEyePoses(session, frameIndex, ovrTrue, HmdToEyePose, EyeRenderPose, &sensorSampleTime);

		// Render Scene to Eye Buffers
		for (int eye = 0; eye < 2; ++eye) {
			// Switch to eye render target
			eyeRenderTexture[eye]->SetAndClearRenderSurface(eyeDepthBuffer[eye]);

			// Do mystical/magical OVR -> XGL fiddly bits for camera translation
			// These were empirically derived from a LOT of experimentation.
			TransformEye(eye);

			// render XGL scene 
			pXgl->XGL::Display();

			eyeRenderTexture[eye]->UnsetRenderSurface();

			// Commit changes to the textures so they get picked up for this rendering frame
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
