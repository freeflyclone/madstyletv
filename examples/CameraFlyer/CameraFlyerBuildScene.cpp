/**************************************************************
** CameraFlyerBuildScene.cpp
**
** Just to demonstrate instantiation of a "ground"
** plane and a single triangle, with default camera manipulation
** via keyboard and mouse.
**************************************************************/
#include "ExampleXGL.h"

class CameraFlyer : public XGLSled {
public:
	CameraFlyer(XGL* xgl) : XGLSled(false) {
		// use the left stick to control yaw, right stick to control pitch & roll of the sled (typical R/C transmitter layout)
		// XGLSled::SampleInput(float yaw, float pitch, float roll) also calls XGLSled::GetFinalMatrix()
		xgl->AddProportionalFunc("Xbox360Controller0", [this, xgl](float v) { SampleInput(-v, 0.0f, 0.0f); SetCamera(xgl); });
		xgl->AddProportionalFunc("Xbox360Controller2", [this, xgl](float v) { SampleInput(0.0f, 0.0f, v); SetCamera(xgl); });
		xgl->AddProportionalFunc("Xbox360Controller3", [this, xgl](float v) { SampleInput(0.0f, -v, 0.0f); SetCamera(xgl); });

		// move sled with Xbox360 controller left & right triggers
		xgl->AddProportionalFunc("Xbox360Controller4", [this, xgl](float v) { MoveFunc(xgl,v); });
		xgl->AddProportionalFunc("Xbox360Controller5", [this, xgl](float v) { MoveFunc(xgl,-v); });
	}

	// lambda function that moves the sled, to be called by 2 proportional axis callbacks
	void MoveFunc(XGL* xgl, float v) {
		glm::vec4 f = glm::toMat4(o) * glm::vec4(0.0, v / 10.0f, 0.0, 0.0);
		p += glm::vec3(f);
		SetCamera(xgl);
	};

	void SetCamera(XGL* xgl)
	{
		glm::vec3 f = glm::toMat3(o) * glm::vec3(0,1,0);
		glm::vec3 u = glm::toMat3(o) * glm::vec3(0,0,1);
		xgl->camera.Set(p, f, u);
	};

};

void ExampleXGL::BuildScene() {
	CameraFlyer *cameraFlyer;

	AddShape("shaders/000-simple", [&](){ cameraFlyer = new CameraFlyer(this); return cameraFlyer; });
}
