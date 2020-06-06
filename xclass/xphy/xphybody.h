#ifndef XPHYBODY_H
#define XPHYBODY_H

#include "xphy.h"

class XPhyBody {
public:
	void SetMatrix();

	XPhyMass m{ 0 };
	XPhySpeed s{ 0 };
	XPhyPoint p{ 0,0,0 };
	XPhyVelocity v{ 0,0,0 };
	XPhyOrientation o;
	XPhyOrientationMatrix model;
};

#endif