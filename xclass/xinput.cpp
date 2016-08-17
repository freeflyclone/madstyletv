#include "xinput.h"

void XInput::KeyEvent(int key, int flags){
	XInputKeyMap::iterator perKey;
	XInputKeyRangeMap::iterator perKeyRange;

	for (perKey = keyMap.begin(); perKey != keyMap.end(); perKey++)
		if (key == perKey->first)
			perKey->second(key, flags);

	for (perKeyRange = keyRangeMap.begin(); perKeyRange != keyRangeMap.end(); perKeyRange++)
		if ((key >= perKeyRange->first.first) && (key <= perKeyRange->first.second))
			perKeyRange->second(key, flags);
}

void XInput::AddKeyFunc(int key, XInputKeyFunc f) {
	keyMap.emplace(key, f);
}

void XInput::AddKeyFunc(XInputKeyRange r, XInputKeyFunc f){
	keyRangeMap.emplace(r, f);
}

void XInput::MouseEvent(int x, int y, int flags){
	XInputMouseFuncs::iterator perFunc;
	for (perFunc = mouseFuncs.begin(); perFunc != mouseFuncs.end(); perFunc++)
		(*perFunc)(x, y, flags);
}

void XInput::AddMouseFunc(XInputMouseFunc f) {
	mouseFuncs.push_back(f);
}