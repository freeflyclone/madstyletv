#include "XGLPhysX.h"

XGLPhysX::XGLPhysX(XGL* px) : pXgl(px), dynamicsSerialNumber(-1), activeActor(NULL), mouseSphere(NULL), mouseJoint(NULL) {
	initPhysics(true);

	pXgl->AddPreRenderFunction([&](float clock) {
		// currently this gets called once for each eye if HMD is running,
		// so only step physics if the "clock" actually changed.
		if (pXgl->useHmd) {
			static float previousClock = 0.0f;
			if (clock != previousClock) {
				stepPhysics(true);
				previousClock = clock;
			}
		}
		else {
			stepPhysics(true);
		}
	});
}

void XGLPhysX::initPhysics(bool interactive)
{
	using namespace std::placeholders;  // for _1, _2, _3...
	mFoundation = PxCreateFoundation(PX_PHYSICS_VERSION, mAllocator, mErrorCallback);
	if (!mFoundation)
		throwXGLException("PxCreateFoundation() failed\n");

	mPvd = physx::PxCreatePvd(*mFoundation);
	physx::PxPvdTransport* transport = physx::PxDefaultPvdSocketTransportCreate("localhost", 5425, 10);
	mPvd->connect(*transport, physx::PxPvdInstrumentationFlag::eALL);


	mPhysics = PxCreatePhysics(PX_PHYSICS_VERSION, *mFoundation, physx::PxTolerancesScale(), false, mPvd);
	if (!mPhysics)
		throwXGLException("PxCreatePhysics() failed");

	PxInitExtensions(*mPhysics, mPvd);

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

	mMaterial = mPhysics->createMaterial(0.5f, 0.5f, 0.5f);

	physx::PxRigidStatic* groundPlane = physx::PxCreatePlane(*mPhysics, physx::PxPlane(0, 0, 1, 0), *mMaterial);
	mScene->addActor(*groundPlane);

	// XGLShape-derived object, for rendering the PhysX primitives.
	// NOTE: this bit of code feels sketchy re: "this" inside Init() not
	// being fully defined yet because of constructors of derived classes.
	// (although I'm not sure,  need an in-depth guru meditation on the subject)
	std::string shaderName = "shaders/specular";
	pXgl->AddShape(shaderName, [&]() { renderer = new PhysxRenderer(this); return renderer; });
	renderer->Init(pXgl->GetShader(shaderName));

	//if(false)
	for (int i = -20; i < 20; i += 5) {
		createChain(physx::PxTransform(physx::PxVec3(float(i), 0.0f, 20.0f)), 10, physx::PxSphereGeometry(1), 1.0f,
			[this](physx::PxRigidActor* a0, const physx::PxTransform& t0, physx::PxRigidActor* a1, const physx::PxTransform& t1){
			physx::PxSpring spring = { 4.0, 4.0 };
			physx::PxSphericalJoint* j = PxSphericalJointCreate(*mPhysics, a0, t0, a1, t1);
			//j->setLimitCone(PxJointLimitCone(PxPi / 2, PxPi / 2, 0.05f));
			j->setLimitCone(physx::PxJointLimitCone(physx::PxPi / 2, physx::PxPi / 2, spring));
			j->setSphericalJointFlag(physx::PxSphericalJointFlag::eLIMIT_ENABLED, true);
			j->setProjectionLinearTolerance(0.1f);
			j->setConstraintFlag(physx::PxConstraintFlag::ePROJECTION, true);
			return j;
		});
	}
}

void  XGLPhysX::createChain(const physx::PxTransform& t, physx::PxU32 length, const physx::PxGeometry& g, physx::PxReal separation, PhysxCreateJointFn createJoint)
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
physx::PxRigidDynamic* XGLPhysX::createDynamic(const physx::PxTransform& t, const physx::PxGeometry& g, physx::PxMaterial& m, float d) {
	physx::PxRigidDynamic *rd = PxCreateDynamic(*mPhysics, t, g, m, d);

	UserData *ud = new UserData(++dynamicsSerialNumber);
	rd->userData = ud;

	return rd;
}

physx::PxRigidDynamic* XGLPhysX::createDynamic(const physx::PxTransform& t, const physx::PxGeometry& g, const physx::PxVec3& velocity) {
	physx::PxRigidDynamic* dynamic = createDynamic(t, g, *mMaterial, 1.0f);
	dynamic->setAngularDamping(0.5f);
	dynamic->setLinearDamping(0.5f);
	dynamic->setLinearVelocity(velocity);
	mScene->addActor(*dynamic);
	return dynamic;
}

physx::PxRigidDynamic* XGLPhysX::CreateDynamicSphere(float radius, const XPhyPoint& p, const XPhyVelocity& v) {
	physx::PxVec3 pos(p.x, p.y, p.z);
	physx::PxTransform t(pos);
	physx::PxVec3 velocity(v.x, v.y, v.z);

	physx::PxSphereGeometry g(radius);
	physx::PxRigidDynamic* prd = PxCreateDynamic(*mPhysics, t, g, *mMaterial, radius);

	if (!prd)
		return nullptr;

	prd->setAngularDamping(0.5f);
	prd->setLinearDamping(0.5f);
	prd->setLinearVelocity(velocity);
	mScene->addActor(*prd);

	UserData* pud = new UserData(++dynamicsSerialNumber);
	pud->shape = new XGLSphere(radius, 32);

	pud->shape->Load(renderer->shader, pud->shape->v, pud->shape->idx);
	pud->shape->uniformLocations = renderer->shader->materialLocations;

	prd->userData = pud;

	return prd;
}

void XGLPhysX::createStack(const physx::PxTransform& t, physx::PxU32 size, physx::PxReal halfExtent) {
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

void XGLPhysX::stepPhysics(bool interactive) {
	PX_UNUSED(interactive);
	if (pXgl->useHmd)
		mScene->simulate(1.0f / 90.0f);
	else
		mScene->simulate(1.0f / 60.0f);

	mScene->fetchResults(true);
}
void XGLPhysX::RenderActors(physx::PxRigidActor** actors, const physx::PxU32 numActors, bool shadows, const physx::PxVec3 & color) {
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
		UserData *pud = (UserData*)actor->userData;

		//bool sleeping = actor->isRigidDynamic() ? actor->isRigidDynamic()->isSleeping() : false;
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
			renderGeometry(h, shapePose, isHit, pud);
		}
	}
}
void XGLPhysX::renderGeometry(physx::PxGeometryHolder h, physx::PxMat44 shapePose, bool isHit, UserData* pud) {
	if (pud && pud->shape) {
		pud->shape->XGLBuffer::Bind();
		pud->shape->XGLMaterial::Bind(pud->shape->shader->programId);
		glProgramUniformMatrix4fv(renderer->shader->programId, renderer->shader->modelUniformLocation, 1, false, (GLfloat *)&shapePose);
		pud->shape->Draw();
		pud->shape->XGLBuffer::Unbind();
	}
	else {
		switch (h.getType())
		{
		case physx::PxGeometryType::eBOX:
			renderer->box->XGLBuffer::Bind();
			renderer->box->attributes.diffuseColor = XGLColors::cyan;
			renderer->box->XGLMaterial::Bind(renderer->box->shader->programId);

			glProgramUniformMatrix4fv(renderer->shader->programId, renderer->shader->modelUniformLocation, 1, false, (GLfloat *)&shapePose);
			glDrawElements(GL_TRIANGLE_STRIP, (GLsizei)(renderer->box->idx.size()), XGLIndexType, 0);

			renderer->box->XGLBuffer::Unbind();
			break;

		case physx::PxGeometryType::eSPHERE:
			renderer->ball->XGLBuffer::Bind();
			renderer->ball->attributes.diffuseColor = XGLColors::yellow;
			renderer->ball->XGLMaterial::Bind(renderer->ball->shader->programId);

			glProgramUniformMatrix4fv(renderer->shader->programId, renderer->shader->modelUniformLocation, 1, false, (GLfloat *)&shapePose);
			glDrawElements(GL_TRIANGLE_STRIP, (GLsizei)(renderer->ball->idx.size()), XGLIndexType, 0);

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
}

void XGLPhysX::PhysxRenderer::Draw()
{
	physx::PxScene* scene;
	PxGetPhysics().getScenes(&scene, 1);
	physx::PxU32 nbActors = scene->getNbActors(physx::PxActorTypeFlag::eRIGID_DYNAMIC | physx::PxActorTypeFlag::eRIGID_STATIC);
	if (nbActors) {
		std::vector<physx::PxRigidActor*> actors(nbActors);
		scene->getActors(physx::PxActorTypeFlag::eRIGID_DYNAMIC | physx::PxActorTypeFlag::eRIGID_STATIC, (physx::PxActor**)&actors[0], nbActors);
		container->RenderActors(&actors[0], (physx::PxU32)actors.size(), true);
	}
}

void XGLPhysX::onShapeHit(const physx::PxControllerShapeHit& hit) {
	xprintf("onShapeHit()\n");
};
void XGLPhysX::onControllerHit(const physx::PxControllersHit& hit) {
	xprintf("onControllerHit()\n");
};
void XGLPhysX::onObstacleHit(const physx::PxControllerObstacleHit& hit) {
	xprintf("onObstacleHit()\n");
};

void XGLPhysX::RayCast(glm::vec3 o, glm::vec3 d) {
	physx::PxVec3 orig(o.x, o.y, o.z);
	physx::PxVec3 dir(d.x, d.y, d.z);
	physx::PxReal length = 1000.0;
	physx::PxRaycastHit hitBuffer[2];
	physx::PxRaycastHit hit;
	physx::PxRaycastBuffer buff(hitBuffer, 2);
	physx::PxActor *actor;

	if (activeActor == NULL) {
		dir -= orig;
		dir.normalize();

		// This has been modified slightly for PhysX 4.0, to make the final argument
		// to raycast compatible with that API version. Without being *completely* sure,
		// I've observed there's always at least one actor that's been hit. Not sure
		// if it's the mouseSphere or the ground plane, but it doesn't seem to matter.
		// Using the second actor in the list of actors returned by raycast() is the
		// actor we want to actually manipulate.
		mScene->raycast(orig, dir, length, buff);
		if (buff.nbTouches > 1) {
			hit = buff.touches[1];
		}

		if ((buff.nbTouches > 1) && (actor = hit.actor) != NULL) {
			if (actor->is<physx::PxRigidDynamic>()) {
				activeActor = actor;
				((UserData *)actor->userData)->active = true;
				hitId = ((UserData *)actor->userData)->id;
				if (mouseSphere == NULL)
				{
					physx::PxVec3 pos(hit.position), dir(0);
					mouseSphere = createDynamic(physx::PxTransform(pos), physx::PxSphereGeometry(0.0001f), dir);
					mouseSphere->setRigidBodyFlag(physx::PxRigidBodyFlag::eKINEMATIC, true);
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

void XGLPhysX::ResetActive() {
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

physx::PxMat44 glmMat4ToPhysxMat4(const glm::mat4& mat4)
{
	physx::PxMat44 newMat;

	newMat[0][0] = mat4[0][0];
	newMat[0][1] = mat4[0][1];
	newMat[0][2] = mat4[0][2];
	newMat[0][3] = mat4[0][3];

	newMat[1][0] = mat4[1][0];
	newMat[1][1] = mat4[1][1];
	newMat[1][2] = mat4[1][2];
	newMat[1][3] = mat4[1][3];

	newMat[2][0] = mat4[2][0];
	newMat[2][1] = mat4[2][1];
	newMat[2][2] = mat4[2][2];
	newMat[2][3] = mat4[2][3];

	newMat[3][0] = mat4[3][0];
	newMat[3][1] = mat4[3][1];
	newMat[3][2] = mat4[3][2];
	newMat[3][3] = mat4[3][3];

	return newMat;
}

typedef struct { physx::PxU32 i1, i2, i3; } TRIANGLE;

void XGLPhysX::ShapeToActor(XGLShape *s) {
	int size = (int)(s->v.size());
	physx::PxVec3* points = (physx::PxVec3*)malloc(size*sizeof(physx::PxVec3));

	XGLIndexList::iterator idx;
	int i;
	physx::PxTriangleMeshDesc md;
	TRIANGLE* tri = (TRIANGLE *)malloc(size * 3 * sizeof(TRIANGLE));

	i = 0;
	for (auto vrtx : s->v) {
		points[i].x = vrtx.v.x;
		points[i].y = vrtx.v.y;
		points[i].z = vrtx.v.z;
		i++;
	}

	i = 0;

	for (idx = s->idx.begin(); idx != s->idx.end() - 2; idx++)
	{
		if ((i & 1) == 0) {
			tri[i].i1 = *idx;
			tri[i].i2 = *(idx + 1);
			tri[i].i3 = *(idx + 2);
			i++;
		}
		else {
			tri[i].i1 = *idx;
			tri[i].i2 = *(idx + 2);
			tri[i].i3 = *(idx + 1);
			i++;
		}
	}

	md.points.count = size;
	md.points.stride = sizeof(physx::PxVec3);
	md.points.data = points;
	md.triangles.count = i;
	md.triangles.stride = sizeof(TRIANGLE);
	md.triangles.data = tri;

	physx::PxDefaultMemoryOutputStream buf;
	bool status = gCooking->cookTriangleMesh(md, buf);
	if (!status)
		throwXGLException("cookTriangleMesh() failed");

	physx::PxDefaultMemoryInputData rbuf(buf.getData(), buf.getSize());

	physx::PxTriangleMesh *triMesh = mPhysics->createTriangleMesh(rbuf);
	if (triMesh == NULL)
		throwXGLException("createTriangleMesh() failed");

	physx::PxTriangleMeshGeometry tmg(triMesh);

	physx::PxShape* shape = mPhysics->createShape(tmg, *mMaterial);
	if (!shape)
		throwXGLException("createShape() failed");

	physx::PxRigidStatic *stat = PxCreateStatic(*mPhysics, physx::PxTransform(glmMat4ToPhysxMat4(s->model)), *shape);
	if (!stat)
		throwXGLException("PxCreateStatic() failed");

	mScene->addActor(*stat);
}
