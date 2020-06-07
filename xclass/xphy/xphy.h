#ifndef XPHY_H
#define XPHY_H

#include "glm.hpp"
#include "glm/gtx/quaternion.hpp"
#include "matrix_transform.hpp"
#include "type_ptr.hpp"

typedef float XPhyMass;
typedef float XPhySpeed;			// meters/sec
typedef float XPhyMagnitude;

typedef glm::vec3 XPhyDirection;	// direction of travel
typedef glm::vec3 XPhyPoint;
typedef glm::vec3 XPhyVelocity;
typedef glm::fquat XPhyOrientation;

struct XPhyForce {
	XPhyDirection d;
	XPhyMagnitude m;
};

typedef glm::mat4 XPhyOrientationMatrix;
#endif