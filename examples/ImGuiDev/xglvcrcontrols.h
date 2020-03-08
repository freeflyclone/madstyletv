#pragma once
#include "ExampleXGL.h"
#include "xglimgui.h"

class XGLVcrControlsGui : public XGLImGui
{
public:
	bool vcrWindow{ true };
	int frameNum{ 0 };
};
