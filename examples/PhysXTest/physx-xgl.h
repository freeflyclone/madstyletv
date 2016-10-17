/****************************************************************************
** Copyright (C) 2016 Evan Mortimore
** All rights reserved. (for now)
**
** Description:
**	Generic application wrapper for xclass, XGL, and XInput application
**  example, with PhysX integration.
*****************************************************************************/
#pragma once

//#include <ctype.h>
#include <PxPhysicsAPI.h>
#include <ExampleXGL.h>

#ifdef _DEBUG
#pragma comment(lib, "PhysX3DEBUG_x64.lib")
#pragma comment(lib, "PhysX3ExtensionsDEBUG.lib")
#pragma comment(lib, "PhysX3CommonDEBUG_x64.lib")
#pragma comment(lib, "PhysX3CookingDEBUG_x64.lib")
#pragma comment(lib, "PhysX3CharacterKinematicDEBUG_x64.lib")
#pragma comment(lib, "PhysX3VehicleDEBUG.lib")
#pragma comment(lib, "PhysXProfileSDKDEBUG.lib")
#pragma comment(lib, "PhysXVisualDebuggerSDKDEBUG.lib")
#pragma comment(lib, "PxTaskDEBUG.lib")
#else
#pragma comment(lib, "PhysX3Extensions.lib")
#pragma comment(lib, "PhysX3Common_x64.lib")
#pragma comment(lib, "PhysX3_x64.lib")
#pragma comment(lib, "PhysX3Cooking_x64.lib")
#pragma comment(lib, "PhysX3CharacterKinematic_x64.lib")
#pragma comment(lib, "PhysX3Vehicle.lib")
#pragma comment(lib, "PhysXProfileSDK.lib")
#pragma comment(lib, "PhysXVisualDebuggerSDK.lib")
#pragma comment(lib, "PxTask.lib")
#endif

#define MAX_NUM_ACTOR_SHAPES 128

// define a bindable function pointer for various Physx Joint types creation
typedef std::function<physx::PxJoint *(physx::PxRigidActor *, const physx::PxTransform&, physx::PxRigidActor*, const physx::PxTransform&)> MstvPhysxCreateJointFunk;

class PhysXXGL : public ExampleXGL, public physx::PxUserControllerHitReport {
public:
	PhysXXGL();
	void BuildScene();

	physx::PxDefaultAllocator			mAllocator;
	physx::PxDefaultErrorCallback		mErrorCallback;
	physx::PxFoundation*				mFoundation;
	physx::PxPhysics*					mPhysics;
	physx::PxDefaultCpuDispatcher*		mDispatcher;
	physx::PxScene*						mScene;
	physx::PxMaterial*					mMaterial;
	physx::PxVisualDebuggerConnection*	mConnection;
	physx::PxControllerManager*			mControllerManager;
	physx::PxController*				mController;

	physx::PxReal						stackY = 10.f;
	physx::PxCooking*					gCooking;

	physx::PxRigidDynamic *createDynamic(const physx::PxTransform& t, const physx::PxGeometry& g, const physx::PxVec3& velocity = physx::PxVec3(0));
	physx::PxRigidDynamic *createDynamic(const physx::PxTransform &, const physx::PxGeometry &, physx::PxMaterial &, float);

	void createStack(const physx::PxTransform& t, physx::PxU32 size, physx::PxReal halfExtent);

	// define functions for creating various types of Physx Joints
	physx::PxJoint* createLimitedSpherical(physx::PxRigidActor*, const physx::PxTransform&, physx::PxRigidActor*, const physx::PxTransform&);
	physx::PxJoint* createBreakableFixed(physx::PxRigidActor*, const physx::PxTransform&, physx::PxRigidActor*, const physx::PxTransform&);
	physx::PxJoint* createDampedD6(physx::PxRigidActor*, const physx::PxTransform&, physx::PxRigidActor*, const physx::PxTransform&);

	// create a chain of Physx Joints of various styles.  This could probably be expanded
	void createChain(const physx::PxTransform&, physx::PxU32, const physx::PxGeometry&, physx::PxReal, MstvPhysxCreateJointFunk);

	void initPhysics(bool interactive);
	void stepPhysics(bool interactive);
	void cleanupPhysics(bool interactive);
	void renderGeometry(physx::PxGeometryHolder, physx::PxMat44, bool);

	void RenderActors(physx::PxRigidActor** actors, const physx::PxU32 numActors, bool shadows = false, const physx::PxVec3 & color = physx::PxVec3(0.0f, 0.75f, 0.0f));

	void ShapeToActor(XGLShape *s);

	void onShapeHit(const physx::PxControllerShapeHit& hit);
	void onControllerHit(const physx::PxControllersHit& hit);
	void onObstacleHit(const physx::PxControllerObstacleHit& hit);

	void RayCast(glm::vec3, glm::vec3);
	void ResetActive();

	class PhysxRenderer : public XGLShape {
	public:
		PhysxRenderer(PhysXXGL *p) : container(p), prevClock(0.0f) {
			v.push_back({ { 0, 0, 0 } });
			box = new XGLCube();
			ball = new XGLSphere(1.0, 36);
			capsule = new XGLCapsule(1.0, 2.0, 36);
		};
		PhysXXGL *container;
		void Draw();
		void Init(XGLShader *shader) {
			box->Load(shader, box->v, box->idx);
			box->SetColor(blue);
			box->l = shader->materialLocations;
			ball->Load(shader, ball->v, ball->idx);
			ball->SetColor(yellow);
			ball->l = shader->materialLocations;
			capsule->Load(shader, capsule->v, capsule->idx);
			capsule->l = shader->materialLocations;
		}

		XGLCube *box;
		XGLSphere *ball;
		XGLCapsule *capsule;
		GLuint program;
		float prevClock;
	};

	class UserData {
	public:
		UserData(int i) : active(false), id(i) {};
		bool active;
		int id;
	};

	PhysxRenderer *renderer;
	physx::PxActor *activeActor;

	int dynamicsSerialNumber;
	int hitId;

	physx::PxDistanceJoint *mouseJoint = NULL;
	physx::PxRigidDynamic *mouseSphere = NULL;
	float mouseDepth;
	physx::PxRigidDynamic *selectedActor = NULL;
};
