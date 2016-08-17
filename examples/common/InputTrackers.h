/****************************************************************************
** Copyright (C) 2016 Evan Mortimore
** All rights reserved. (for now)
**
** Description:
**	Input tracking involves keeping state for the keyboard and mouse.
**
**  The low-level XInput class abstracts the OS specific code for the
**	keyboard and mouse devices.
**
**	The decision of HOW to use the input events is application specific:
**  a 1st-person shooter  might want to move the camera according to W,A,S,D
**  keys for example.  The mouse might move the camera, or an object in the
**	game, or whatever. (these are what this example does)
**
**  I'm not clear what's the best way to cleanly abstract the 
**  events -> actions mappings, so I'm opting for a simple example instead.
**
**  In the end, I feel it's probably up to the client application developer
**  to choose his/her preferences.
*****************************************************************************/
#pragma once

class KeyboardTracker {
public:
	KeyboardTracker() :f(false), b(false), r(false), l(false), wu(false), wd(false) {};
	bool f, b, r, l, wu, wd;
};

class MouseTracker {
public:
	MouseTracker() : l(false), r(false), m(false) {};

	bool IsTracking() { return tracking; }
	bool IsTrackingLeftButton() { return tracking && l; }
	bool IsTrackingRightButton() { return tracking && r; }

	void Event(int x, int y, int flags) {
		float fx = (float)x;
		float fy = (float)y;
		l = (flags & 1) != 0;
		r = (flags & 2) != 0;

		if ((l || r) && !tracking) {
			tracking = true;
			anchorX = fx;
			anchorY = fy;
		}

		if (tracking && !(l || r))
			tracking = false;

		if (tracking) {
			dx = fx - anchorX;
			dy = fy - anchorY;
			anchorX = fx;
			anchorY = fy;
		}
	};

	void Done() {
		dx = 0;
		dy = 0;
	}

	bool l, m, r, tracking;
	float anchorX, anchorY, dx, dy;
};