#include "physx-xgl.h"

PhysXXGL::PhysXXGL() {
	xprintf("PhysXXGL::PhysXXGL()\n");

	//initPhysics(true);
}

void PhysXXGL::initPhysics(bool interactive)
{
	using namespace std::placeholders;  // for _1, _2, _3...
	mFoundation = PxCreateFoundation(PX_PHYSICS_VERSION, mAllocator, mErrorCallback);
	if (!mFoundation)
		throwXGLException("PxCreateFoundation() failed\n");

	physx::PxProfileZoneManager* profileZoneManager = &physx::PxProfileZoneManager::createProfileZoneManager(mFoundation);
	if (!profileZoneManager)
		throwXGLException("createProfileZoneManager() failed\n");

	mPhysics = PxCreatePhysics(PX_PHYSICS_VERSION, *mFoundation, physx::PxTolerancesScale(), true, profileZoneManager);
	if (!mPhysics)
		throwXGLException("PxCreatePhysics() failed");

	PxInitExtensions(*mPhysics);

	// Added by me to allow for cooking a triangle mesh, which I use for collision detection with XGLShape objects
	// in ShapeToActor().  This is *supposed* to be able to work with any triangle mesh from XGLShape class,
	// but so far it's only been tested on XGLTorus
	gCooking = PxCreateCooking(PX_PHYSICS_VERSION, *mFoundation, physx::PxCookingParams(mPhysics->getTolerancesScale()));
	if (!gCooking)
		throwXGLException("PxCreateCooking() failed");

	physx::PxSceneDesc sceneDesc(mPhysics->getTolerancesScale());
	sceneDesc.gravity = physx::PxVec3(0.0f, 0.0f, -9.81f);
	mDispatcher = physx::PxDefaultCpuDispatcherCreate(2);
	sceneDesc.cpuDispatcher = mDispatcher;
	sceneDesc.filterShader = physx::PxDefaultSimulationFilterShader;
	mScene = mPhysics->createScene(sceneDesc);

	mMaterial = mPhysics->createMaterial(0.9f, 0.9f, 0.01f);

	physx::PxRigidStatic* groundPlane = physx::PxCreatePlane(*mPhysics, physx::PxPlane(0, 0, 1, 0), *mMaterial);
	mScene->addActor(*groundPlane);

	AddShape("shaders/diffuse", [&]() { renderer = new PhysxRenderer(this); return renderer; });
	renderer->program = shaderMap[pathToAssets + "/shaders/diffuse"]->Id();
	renderer->ball->program = renderer->program;
	renderer->box->program = renderer->program;
	renderer->Init();

	//if(false)
	for (int i = -20; i < 20; i += 5) {
		//createChain(PxTransform(PxVec3(float(i), 0.0f, 20.0f)), 10, PxBoxGeometry(1, 1, 1), 2.0f, std::bind(&MstvPhysx::createLimitedSpherical, this, _1, _2, _3, _4));
		createChain(physx::PxTransform(physx::PxVec3(float(i), 0.0f, 20.0f)), 10, physx::PxSphereGeometry(1), 1.0f, std::bind(&PhysXXGL::createLimitedSpherical, this, _1, _2, _3, _4));
	}
}

physx::PxJoint* PhysXXGL::createLimitedSpherical(physx::PxRigidActor* a0, const physx::PxTransform& t0, physx::PxRigidActor* a1, const physx::PxTransform& t1)
{
	physx::PxSpring spring = { 4.0, 4.0 };
	physx::PxSphericalJoint* j = PxSphericalJointCreate(*mPhysics, a0, t0, a1, t1);
	//j->setLimitCone(PxJointLimitCone(PxPi / 2, PxPi / 2, 0.05f));
	j->setLimitCone(physx::PxJointLimitCone(physx::PxPi / 2, physx::PxPi / 2, spring));
	j->setSphericalJointFlag(physx::PxSphericalJointFlag::eLIMIT_ENABLED, true);
	j->setProjectionLinearTolerance(0.1f);
	j->setConstraintFlag(physx::PxConstraintFlag::ePROJECTION, true);
	return j;
}

void  PhysXXGL::createChain(const physx::PxTransform& t, physx::PxU32 length, const physx::PxGeometry& g, physx::PxReal separation, MstvPhysxCreateJointFunk createJoint)
{
	physx::PxVec3 offset(0, separation / 2, 0);
	physx::PxTransform localTm(offset);

	// if the initialization of "prev" is NULL, the joint will be "fixed" to "t"
	//PxRigidDynamic* prev = NULL;
	physx::PxRigidDynamic* prev = createDynamic(t*localTm, g, *mMaterial, 1.0f);
	for (physx::PxU32 i = 0; i<length; i++)
	{
		physx::PxRigidDynamic* current = createDynamic(t*localTm, g, *mMaterial, 1.0f);
		createJoint(prev, prev ? physx::PxTransform(offset) : t, current, physx::PxTransform(-offset));
		mScene->addActor(*current);
		prev = current;
		localTm.p.x += separation;
	}
}
physx::PxRigidDynamic* PhysXXGL::createDynamic(const physx::PxTransform& t, const physx::PxGeometry& g, physx::PxMaterial& m, float d) {
	physx::PxRigidDynamic *rd = PxCreateDynamic(*mPhysics, t, g, m, d);

	rd->userData = new UserData(++dynamicsSerialNumber);

	return rd;
}

void PhysXXGL::stepPhysics(bool interactive)
{
	PX_UNUSED(interactive);
	mScene->simulate(1.0f / 60.0f);
	mScene->fetchResults(true);
}
void PhysXXGL::RenderActors(physx::PxRigidActor** actors, const physx::PxU32 numActors, bool shadows, const physx::PxVec3 & color) {
	physx::PxShape* shapes[MAX_NUM_ACTOR_SHAPES];
	physx::PxRigidActor *actor;

	// this for loop is evolving: I'm copying bits from the Snippets source piecemeal for better understanding.
	for (physx::PxU32 i = 0; i<numActors; i++)	{
		// this block is straight from the source.
		const physx::PxU32 nbShapes = actors[i]->getNbShapes();
		PX_ASSERT(nbShapes <= MAX_NUM_ACTOR_SHAPES);
		actor = actors[i];
		if (actor == mouseSphere)
			continue;
		actor->getShapes(shapes, nbShapes);
		bool sleeping = actor->isRigidDynamic() ? actor->isRigidDynamic()->isSleeping() : false;
		bool isHit = false;

		if (UserData *ud = (UserData *)actor->userData) {
			if (ud->active)
				isHit = true;
		}

		// this block is incomplete - consult the original file!
		for (physx::PxU32 j = 0; j<nbShapes; j++) {
			const physx::PxMat44 shapePose(physx::PxShapeExt::getGlobalPose(*shapes[j], *actors[i]));
			physx::PxGeometryHolder h = shapes[j]->getGeometry();

			// rendering the geometry
			renderGeometry(h, shapePose, isHit);
		}
	}
}
void PhysXXGL::renderGeometry(physx::PxGeometryHolder h, physx::PxMat44 shapePose, bool isHit) {
	switch (h.getType())
	{
		case physx::PxGeometryType::eBOX:
			renderer->box->XGLBuffer::Bind();
			renderer->box->XGLBuffer::Unbind();
			break;

		case physx::PxGeometryType::eSPHERE:
			renderer->ball->XGLBuffer::Bind();
			renderer->ball->diffuseColor = XGLColor(1, 1, 0);
			renderer->ball->XGLMaterial::Bind(renderer->ball->program);

			glBufferSubData(GL_UNIFORM_BUFFER, sizeof(glm::mat4) * 2, sizeof(glm::mat4), (GLvoid *)&shapePose);
			//GL_CHECK("glBufferSubData() failed");
			glDrawElements(GL_TRIANGLE_STRIP, (GLsizei)(renderer->ball->idx.size()), XGLIndexType, 0);
			//GL_CHECK("glDrawElements() failed");

			renderer->ball->XGLBuffer::Unbind();

			break;

		case physx::PxGeometryType::eCAPSULE:
			break;

		case physx::PxGeometryType::eCONVEXMESH:
			break;

		case physx::PxGeometryType::eTRIANGLEMESH:
			break;

		default:
			break;
	}
}

void PhysXXGL::PhysxRenderer::Draw()
{
	container->stepPhysics(true);

	physx::PxScene* scene;
	PxGetPhysics().getScenes(&scene, 1);
	physx::PxU32 nbActors = scene->getNbActors(physx::PxActorTypeSelectionFlag::eRIGID_DYNAMIC | physx::PxActorTypeSelectionFlag::eRIGID_STATIC);
	if (nbActors) {
		std::vector<physx::PxRigidActor*> actors(nbActors);
		scene->getActors(physx::PxActorTypeSelectionFlag::eRIGID_DYNAMIC | physx::PxActorTypeSelectionFlag::eRIGID_STATIC, (physx::PxActor**)&actors[0], nbActors);
		container->RenderActors(&actors[0], (physx::PxU32)actors.size(), true);
	}
}

void PhysXXGL::onShapeHit(const physx::PxControllerShapeHit& hit) {
	xprintf("onShapeHit()\n");
};
void PhysXXGL::onControllerHit(const physx::PxControllersHit& hit) {
	xprintf("onControllerHit()\n");
};
void PhysXXGL::onObstacleHit(const physx::PxControllerObstacleHit& hit) {
	xprintf("onObstacleHit()\n");
};
