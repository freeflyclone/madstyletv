/**************************************************************
** CameraFlyerBuildScene.cpp
**
** Demonstrates the new XGLCameraFlyer shape.  XBox 360 
** controller flying the XGLCamera as in aircraft cockpit.
**************************************************************/
#include "ExampleXGL.h"

void ExampleXGL::BuildScene() {
	XGLCameraFlyer *cameraFlyer;

	AddShape("shaders/000-simple", [&](){ cameraFlyer = new XGLCameraFlyer(this); return cameraFlyer; });
}

