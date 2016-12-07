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

#include "xutils.h"

using namespace std::placeholders;

class XInput {
public:
	typedef std::function<void(int, int)> XInputKeyFunc;
	typedef std::multimap<char, XInputKeyFunc> XInputKeyMap;

	typedef std::pair<int, int> XInputKeyRange;
	typedef std::map<XInputKeyRange, XInputKeyFunc> XInputKeyRangeMap;

	typedef std::function<void(int, int, int)> XInputMouseFunc;
	typedef std::vector<XInputMouseFunc> XInputMouseFuncs;

	void AddMouseFunc(XInputMouseFunc);

	void AddKeyFunc(int, XInputKeyFunc);
	void AddKeyFunc(XInputKeyRange, XInputKeyFunc);


	void MouseEvent(int, int, int) const;
	void KeyEvent(int, int) const;

private:
	XInputKeyMap keyMap;
	XInputKeyRangeMap keyRangeMap;
	XInputMouseFuncs mouseFuncs;
};
#endif