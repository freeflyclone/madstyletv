/****************************************************************************
**
** Copyright (C) 2016 Evan Mortimore
** All rights reserved.
**
** definitions of XInput objects:
**
****************************************************************************/
#ifndef XINPUT_H
#define XINPUT_H

#include <functional>
#include <map>
#include <vector>
#include <string>

#include "xutils.h"

using namespace std::placeholders;

#ifndef MAX_PATH
#define MAX_PATH 260
#endif

typedef std::function<const float*(int*)> XJoystickAxesPoll;

struct XJoystick {
	char fullName[MAX_PATH];
	char shortName[MAX_PATH];
	int numAxes;
	float values[32]; // doubtful there'll be more than 32 axes on a joystick;
	XJoystickAxesPoll pollFunc;
};
typedef std::vector<XJoystick> XJoysticks;

class XInput {
public:
	typedef std::function<void(int, int)> XInputKeyFunc;
	typedef std::multimap<int, XInputKeyFunc> XInputKeyMap;

	typedef std::pair<int, int> XInputKeyRange;
	typedef std::map<XInputKeyRange, XInputKeyFunc> XInputKeyRangeMap;

	typedef std::function<void(int, int, int)> XInputMouseFunc;
	typedef std::vector<XInputMouseFunc> XInputMouseFuncs;

	typedef std::function<void(float)> XInputProportionalFunc;
	typedef std::multimap<std::string, XInputProportionalFunc> XInputProportionalFuncs;

	void AddMouseFunc(XInputMouseFunc);

	void AddKeyFunc(int, XInputKeyFunc);
	void AddKeyFunc(XInputKeyRange, XInputKeyFunc);

	void AddProportionalFunc(std::string, XInputProportionalFunc);

	void AddJoystick(XJoystick&);
	void PollJoysticks();

	void MouseEvent(int, int, int) const;
	void KeyEvent(int, int) const;
	void ProportionalEvent(std::string, float) const;

private:
	XInputKeyMap keyMap;
	XInputKeyRangeMap keyRangeMap;
	XInputMouseFuncs mouseFuncs;
	XInputProportionalFuncs proportionalMap;
	XJoysticks joysticks;
};
#endif
