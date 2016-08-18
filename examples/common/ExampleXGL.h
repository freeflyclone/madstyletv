/****************************************************************************
** Copyright (C) 2016 Evan Mortimore
** All rights reserved. (for now)
**
** Description:
**	OS-specific (in this case Windows) application wrapper for xclass,
**  XGL, and XInput application example.
**
**  Objective is a thin glue layer between Windows stuff (OpenGL context
**  and input device events (keyboard,mouse)) and the XGL Framework.
*****************************************************************************/
#pragma once
#include "xgl.h"
#include "xinput.h"
#include "InputTrackers.h"

class WinXGL : public XGL, public XInput {
public:
	WinXGL();
	virtual void Display();
	void Reshape(int w, int h);

	void MouseFunc(int x, int y, int flags);
	void KeyFunc(int key, int flags);

	void CameraTracker(XGLCamera *c);

private:
	// application-specific input events -> actions mappings
	MouseTracker mt;
	KeyboardTracker kt;
};