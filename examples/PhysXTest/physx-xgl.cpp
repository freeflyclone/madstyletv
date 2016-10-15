#include "physx-xgl.h"

PhysXXGL::PhysXXGL() : dynamicsSerialNumber(-1), activeActor(NULL), mouseSphere(NULL), mouseJoint(NULL) {
	xprintf("PhysXXGL::PhysXXGL()\n");

	initPhysics(true);

	BuildScene();
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

	// XGLShape-derived object, for rendering the PhysX primitives.
	// NOTE: this bit of code feels sketchy re: "this" inside Init() not
	// being fully defined yet because of constructors of derived classes.
	// (although I'm not sure,  need an in-depth guru meditation on the subject)
	std::string shaderName = "shaders/diffuse";
	AddShape(shaderName, [&]() { renderer = new PhysxRenderer(this); return renderer; });
	renderer->Init(GetShader(shaderName));
	renderer->SetTheFunk([&](XGLShape *s, float clock) {
		if (clock > renderer->prevClock) {
			stepPhysics(true);
			renderer->prevClock = clock;
		}
	});

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

physx::PxRigidDynamic* PhysXXGL::createDynamic(const physx::PxTransform& t, const physx::PxGeometry& geometry, const physx::PxVec3& velocity) {
	physx::PxRigidDynamic* dynamic = createDynamic(t, geometry, *mMaterial, 10.0f);
	dynamic->setAngularDamping(0.5f);
	dynamic->setLinearDamping(0.5f);
	dynamic->setLinearVelocity(velocity);
	mScene->addActor(*dynamic);
	return dynamic;
}

void PhysXXGL::createStack(const physx::PxTransform& t, physx::PxU32 size, physx::PxReal halfExtent) {
	physx::PxShape* shape = mPhysics->createShape(physx::PxBoxGeometry(halfExtent, halfExtent, halfExtent), *mMaterial);

	for (physx::PxU32 i = 3; i<size; i++)
	{
		for (physx::PxU32 j = 0; j<size - i; j++)
		{
			physx::PxTransform localTm(physx::PxVec3(physx::PxReal(j * 2) - physx::PxReal(size - i), 0, physx::PxReal(i * 2 + 1)) * halfExtent);
			physx::PxRigidDynamic* body = createDynamic(t.transform(localTm), physx::PxBoxGeometry(1, 1, 1), *mMaterial, 1.0f);
			body->attachShape(*shape);
			physx::PxRigidBodyExt::updateMassAndInertia(*body, 10.0f);
			mScene->addActor(*body);
		}
	}
	shape->release();
}

void PhysXXGL::stepPhysics(bool interactive) {
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
			renderer->box->m.diffuseColor = XGLColor(0, 1, 1, 1);
			renderer->box->XGLMaterial::Bind(renderer->box->program);

			glBufferSubData(GL_UNIFORM_BUFFER, sizeof(glm::mat4) * 2, sizeof(glm::mat4), (GLvoid *)&shapePose);
			//GL_CHECK("glBufferSubData() failed");
			glDrawElements(GL_TRIANGLE_STRIP, (GLsizei)(renderer->box->idx.size()), XGLIndexType, 0);
			//GL_CHECK("glDrawElements() failed");

			renderer->box->XGLBuffer::Unbind();
			break;

		case physx::PxGeometryType::eSPHERE:
			renderer->ball->XGLBuffer::Bind();
			renderer->ball->m.diffuseColor = XGLColor(1, 1, 0, 1);
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

void PhysXXGL::RayCast(glm::vec3 o, glm::vec3 d) {
	physx::PxVec3 orig(o.x, o.y, o.z);
	physx::PxVec3 dir(d.x, d.y, d.z);
	physx::PxReal length = 1000.0;
	physx::PxRaycastHit hit;
	physx::PxActor *actor;

	if (activeActor == NULL) {
		dir -= orig;
		dir.normalize();
		mScene->raycastSingle(orig, dir, length, physx::PxHitFlag::eDEFAULT, hit);

		if ((actor = hit.actor) != NULL) {
			if (actor->is<physx::PxRigidDynamic>()) {
				activeActor = actor;
				((UserData *)actor->userData)->active = true;
				hitId = ((UserData *)actor->userData)->id;
				if (mouseSphere == NULL)
				{
					physx::PxVec3 pos(hit.position), dir(0);
					mouseSphere = createDynamic(physx::PxTransform(pos), physx::PxSphereGeometry(0.0001f), dir);
					mouseSphere->setRigidDynamicFlag(physx::PxRigidDynamicFlag::eKINEMATIC, true);
					selectedActor = (physx::PxRigidDynamic*)actor;
					selectedActor->wakeUp();

					physx::PxTransform mFrame, sFrame;
					mFrame.q = mouseSphere->getGlobalPose().q;
					mFrame.p = mouseSphere->getGlobalPose().transformInv(hit.position);
					sFrame.q = selectedActor->getGlobalPose().q;
					sFrame.p = selectedActor->getGlobalPose().transformInv(hit.position);

					mouseJoint = PxDistanceJointCreate(*mPhysics, mouseSphere, mFrame, selectedActor, sFrame);
					mouseJoint->setConstraintFlag(physx::PxConstraintFlag::ePROJECTION, true);
					mouseJoint->setDamping(10000.0f);
					mouseJoint->setStiffness(200000.0f);
					mouseJoint->setMinDistance(0.00001f);
					mouseJoint->setMaxDistance(0.0001f);
					mouseJoint->setDistanceJointFlag(physx::PxDistanceJointFlag::eMAX_DISTANCE_ENABLED, true);
					mouseJoint->setDistanceJointFlag(physx::PxDistanceJointFlag::eSPRING_ENABLED, true);

					float dx = hit.position.x - orig.x;
					float dy = hit.position.y - orig.y;
					float dz = hit.position.z - orig.z;
					mouseDepth = sqrt(dx*dx + dy*dy + dz*dz);
				}
				return;
			}
		}
	}
	else {
		physx::PxVec3 d = dir - orig;
		d.normalize();
		d *= mouseDepth;
		d += orig;
		if (d.z < 0.0f)
			d.z = 0.0f;
		physx::PxTransform t(d);
		mouseSphere->setKinematicTarget(t);
	}
}

void PhysXXGL::ResetActive() {
	if (activeActor) {
		physx::PxRigidDynamic *rd = (physx::PxRigidDynamic *)activeActor;
		rd->wakeUp();
		((UserData *)activeActor->userData)->active = false;
	}
	activeActor = NULL;

	if (mouseJoint)
		mouseJoint->release();
	mouseJoint = NULL;

	if (mouseSphere)
		mouseSphere->release();
	mouseSphere = NULL;
}
