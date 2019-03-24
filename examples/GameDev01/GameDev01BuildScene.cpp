/**************************************************************
** GameDev01BuildScene.cpp
**
** Simple Game Demo, with PhysX and VR (Oculus) integration.
** This differs from PhysXTest in that PhysX is integrated
** differently: not using a separate main.cpp file.
**************************************************************/
#include "ExampleXGL.h"
#include "XGLPhysX.h"

extern bool initHmd;
XGLPhysX* px;

void ExampleXGL::BuildScene() {
	initHmd = false;

	px = new XGLPhysX(this);
}
