#ifndef XPHYBODY_H
#define XPHYBODY_H

#include "xphy.h"

class XPhyBody {
public:
	XPhyMass m;
	XPhyPoint p;
	XPhyVelocity v;
	XPhyOrientation o;
	XPhyOrientationMatrix model;
};

#endif