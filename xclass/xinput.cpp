#include "xinput.h"

void XInput::KeyEvent(int key, int flags) const {
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

void XInput::MouseEvent(int x, int y, int flags) const{
	for (const auto perFunc : mouseFuncs)
		perFunc(x, y, flags);
}

void XInput::AddMouseFunc(XInputMouseFunc f) {
	mouseFuncs.push_back(f);
}

void XInput::ProportionalEvent(std::string s, float v) const{
	for (const auto perFunc : proportionalMap)
		if (perFunc.first == s)
			perFunc.second(v);
}

void XInput::AddProportionalFunc(std::string key, XInputProportionalFunc f) {
	proportionalMap.emplace(key, f);
}

void XInput::AddJoystick(XJoystick& j) {
	joysticks.push_back(j);
}

void XInput::PollJoysticks() {
	char keyName[MAX_PATH];

	for (auto joystick : joysticks) {
		int nAxes = 0;

		const float *values = joystick.pollFunc(&nAxes);
		for (int i = 0; i < nAxes; i++) {
			sprintf(keyName, "%s%d", joystick.shortName, i);
			ProportionalEvent(keyName, values[i]);
		}
	}
}