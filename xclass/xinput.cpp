#include "xinput.h"

void XInput::KeyEvent(int key, int flags){
	for (auto perKey : keyMap)
		if (key == perKey.first)
				perKey.second(key, flags);

	for (auto perKeyRange : keyRangeMap)
		if ((key >= perKeyRange.first.first) && (key <= perKeyRange.first.second))
			perKeyRange.second(key, flags);
}

void XInput::AddKeyFunc(int key, XInputKeyFunc f) {
	keyMap.emplace(key, f);
}

void XInput::AddKeyFunc(XInputKeyRange r, XInputKeyFunc f){
	keyRangeMap.emplace(r, f);
}

void XInput::MouseEvent(int x, int y, int flags){
	for (auto perFunc : mouseFuncs)
		perFunc(x, y, flags);
}

void XInput::AddMouseFunc(XInputMouseFunc f) {
	mouseFuncs.push_back(f);
}