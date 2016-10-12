/****************************************************************************
** Copyright (C) 2016 Evan Mortimore
** All rights reserved. (for now)
**
** Description:
**	Generic application wrapper for xclass, XGL, and XInput application 
**  example.
**
**  Objective is a thin glue layer between OS-specific items with their
**  corresponding XGL counter parts.  (OpenGL context and input device events
**  (keyboard,mouse)) and the XGL Framework.  This is the generic side,
**  all application code relating to using OpenGL and the XGL framework
**  should be contained herein.
**
**  This provides basic camera manipulation via mouse and keyboard.
*****************************************************************************/
#pragma once
#include "xgl.h"
#include "xinput.h"
#include "xglworldcursor.h"
#include "InputTrackers.h"

class ExampleXGL : public XGL, public XInput {
public:
	ExampleXGL();

	void BuildScene();

	void Reshape(int w, int h);

	void MouseFunc(int x, int y, int flags);
	void KeyFunc(int key, int flags);

	void CameraTracker(XGLCamera *c);

	// application-specific input events -> actions mappings
	MouseTracker mt;
	KeyboardTracker kt;
	XGLWorldCursor wc;
};