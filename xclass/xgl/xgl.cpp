#include "xgl.h"

XGLFont font;

std::string currentWorkingDir;
std::string pathToAssets;

void CheckGlError(const char *file, int line, std::string what){
    GLenum err = glGetError();
    const GLubyte *errString = gluErrorString(err);
    if (err){
        std::string estr(
            file + 
            std::string(":") + 
            std::to_string(line) + 
            std::string(": ") + 
            what +
            std::string(": ") +
            std::string((char *)errString)
        );
        throwXGLException(estr);
        //DebugPrintf("XGLException:: %s", estr.c_str());
    }
}

void CheckGlStatus(const char *file, int line) {
	GLenum err = glGetError();
	if (err) {
		const GLubyte *errString = gluErrorString(err);
		xprintf("%s:%d - %s\n", file, line, errString);
	}
}

XGL::XGL() : clock(0.0f), pb(NULL), fb(NULL), renderGui(false), guiManager(nullptr), mouseCaptured(nullptr), keyboardFocused(nullptr),
preferredWidth(1280),
preferredHeight(720),
useHmd(false),
preferredSwapInterval(1),
pHmd(nullptr),
hmdSled(nullptr)
{
	SetName("XGL");
	xprintf("OpenGL version: %s\n", glGetString(GL_VERSION));
	glGetError();

	shapeLayers.push_back(new XGLShapesMap());
	shapeLayers.push_back(new XGLShapesMap());
	shapeLayers.push_back(new XGLShapesMap());
	shapeLayers.push_back(new XGLShapesMap());

	//QueryContext();

	// for now, create one point light, for diffuse lighting shader development
	// position, pad1, color, pad2, attenuation, ambientCoefficient
	XGLLight light = { { 10, 6, 16 }, 1.0f, { 1, 1, 1, 1 }, 1.0, 0.001f, 0.005f };
	lights.push_back(light);

	// NOTE: It is not necessary to know the names of uniform blocks within the shader.
	// It IS necessary to know that the GL_UNIFORM_BUFFER target is indexed, and thus
	// glBindBufferBase() must be used to bind a uniform buffer to a particular
	// index.  Using glBindBuffer() with GL_UNIFORM_BUFFER should be avoided
	// because while it will work, it can lead to ambiguity if there is more than one
	// uniform buffer object.
	//
	// The tutorial at 
	//    http://http://www.lighthouse3d.com/tutorials/glsl-tutorial/uniform-blocks/
	// proved to be a bit misleading... the code below is correct, per reading the
	// OpenGL documentation of glBindBufferBase().
	//
	// Also, it is valid to set these up before creating GLSL shader programs which
	// use them.  Uniform blocks are designed to be shared amongst shader programs,
	// therefore their creation is independent from programs.

	// utilize UBO for Matrix data
	{
		glGenBuffers(1, &matrixUbo);
		GL_CHECK("glGenBuffers() failed");

		glBindBufferBase(GL_UNIFORM_BUFFER, 0, matrixUbo);
		GL_CHECK("glBindBufferBase() failed");

		glBufferData(GL_UNIFORM_BUFFER, sizeof(shaderMatrix), &shaderMatrix, GL_DYNAMIC_DRAW);
		GL_CHECK("glBufferData() failed");
	}

	// utilize UBO for lighting parameters.
	{
		glGenBuffers(1, &lightUbo);
		GL_CHECK("glGenBuffers() failed");

		glBindBufferBase(GL_UNIFORM_BUFFER, 1, lightUbo);
		GL_CHECK("glBindBufferBase() failed");

		glBufferData(GL_UNIFORM_BUFFER, sizeof(light), &light, GL_DYNAMIC_DRAW);
		GL_CHECK("glBufferData() failed");
	}

	// force an unbinding of the generic GL_UNIFORM_BUFFER binding point, so that
	// code that comes after can't accidently work by some fluke that goes undeteced.
	// (don't ask me how I know.)
	glBindBuffer(GL_UNIFORM_BUFFER, 0);
	GL_CHECK("glBindBuffer(0) failed");

    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
	glDepthFunc(GL_LESS);
	glEnable(GL_DEPTH_TEST);
	//glEnable(GL_BLEND);
	//glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
	//glEnable(GL_CULL_FACE);
	glCullFace(GL_BACK);

	// for copying to shared memory buffer
	// enabling this takes up GPU time,
	// mostly because of glReadPixels().
	// with the encoder in the loop, it's even worse
	if (config.Find(L"SharedMemory")->AsBool())
		fb = new XGLSharedFBO(this);
}

XGL::~XGL(){
	threadPool.clear();

    // iterate through all of the shapes, according to which shader they use
    XGLShapesMap::iterator perShader;

    // iterate through all shapes associated with a particular shader 
    XGLShapeList::iterator perShape;

    // pointer to an XGLShader from the shaderMap
    XGLShader *shader;

	for (perShader = guiShapes.begin(); perShader != guiShapes.end(); perShader++) {
		std::string name = perShader->first;
		shader = shaderMap[name];

		for (perShape = perShader->second->begin(); perShape != perShader->second->end(); perShape++) {
			XGLShape *shape = *perShape;
			delete shape;
		}
	}
	
	for (auto shapes : shapeLayers) {
		for (perShader = shapes->begin(); perShader != shapes->end(); perShader++) {
			std::string name = perShader->first;
			shader = shaderMap[name];

			for (perShape = perShader->second->begin(); perShape != perShader->second->end(); perShape++) {
				XGLShape *shape = *perShape;
				delete shape;
			}
			delete perShader->second;
		}
	}

	// since we now have layers do NOT delete the shader above as was previously done,
	// wait for all shapes layers to be deleted THEN delete all the shaders.
	for (auto shaderMapEntry : shaderMap)
		delete shaderMapEntry.second;
}
void XGL::InitHmd()	{
#ifdef LINUX
	return;
#else
	// Create a cockpit that can be flown in the world, put it in layer 2 to override world object rendering
	// (Turns out the layers hack only works between top level shapes right now)
	AddShape("shaders/000-simple", [&]() { hmdSled = new XGLSled(); return hmdSled; }, 2);
	hmdSled->SetName("HmdSled", false);

	// move forward/backward
	AddProportionalFunc("LeftThumbStick.y", [this](float v) {
		glm::vec4 backward = glm::toMat4(hmdSled->o) * glm::vec4(0.0, v / 10.0f, 0.0, 0.0);
		hmdSled->p += glm::vec3(backward);
		hmdSled->model = hmdSled->GetFinalMatrix();
	});

	// yaw (rudder)
	AddProportionalFunc("LeftThumbStick.x", [this](float v) { hmdSled->SampleInput(-v, 0.0f, 0.0f); });

	// pitch (elevator)
	AddProportionalFunc("RightThumbStick.y", [this](float v) { hmdSled->SampleInput(0.0f, -v, 0.0f); });

	// roll (ailerons)
	AddProportionalFunc("RightThumbStick.x", [this](float v) { hmdSled->SampleInput(0.0f, 0.0f, v); });

	// change the default configuration so the HMD will work.
	preferredWidth = 1080;
	preferredHeight = 600;

	pHmd = new XGLHmd(this, preferredWidth, preferredHeight);
	useHmd = true;
	preferredSwapInterval = 0;
#endif
}


void XGL::RenderScene(XGLShapesMap *shapes) {
	if (!useHmd) {
		// set the projection,view,orthoProjection matrices in the matrix UBO
		shaderMatrix.view = camera.GetViewMatrix();
		shaderMatrix.projection = projector.GetProjectionMatrix();
		shaderMatrix.orthoProjection = projector.GetOrthoMatrix();
	}

	glBufferData(GL_UNIFORM_BUFFER, sizeof(shaderMatrix), (GLvoid *)&shaderMatrix, GL_DYNAMIC_DRAW);
	GL_CHECK("glBufferData() failed");

	for (auto const perShader : *shapes) {
		const XGLShader *shader = shaderMap[perShader.first];
		
		shader->Use();

		glUniform3fv(glGetUniformLocation(shader->programId, "cameraPosition"), 1, (GLfloat*)glm::value_ptr(camera.pos));
		GL_CHECK("glUniform3fv() failed");

		for (auto const shape : *(perShader.second))
			if (shape->isVisible)
				shape->Render();
		
        shader->UnUse();
    }
}

void XGL::Animate() {
	camera.Animate();

	for (auto shapesMaps : shapeLayers)
		for (auto const shader : *shapesMaps)
			for (auto const shape : *shader.second)
				shape->Animate(clock);

	clock += 1.0f;
}

void XGL::PreRender() {
	for (auto fn : preRenderFunctions)
		fn(clock);
}

void XGL::PostRender() {
	for (auto fn : postRenderFunctions)
		fn(clock);
}

bool XGL::Display(){
	PreRender();

	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	GL_CHECK("glClear() failed");

	glBindBufferBase(GL_UNIFORM_BUFFER, 0, matrixUbo);
	GL_CHECK("glBindBuffer() failed");

	// render the world
	for (auto shapes : shapeLayers)
		RenderScene(shapes);

	// render the GUI
	if (renderGui) {
		glDisable(GL_DEPTH_TEST);
		RenderScene(&guiShapes);
		glEnable(GL_DEPTH_TEST);
	}

	if (fb)
		fb->Render(projector.width, projector.height);

	if (pb)
		pb->Render();

	PostRender();

	// always return shouldQuit = false
	return false;
}

XGLShape* XGL::CreateShape(XGLShapesMap *shapes, std::string shName, XGLNewShapeLambda fn){
	std::string shaderName = pathToAssets + "/" + shName;
	if (shaderMap.count(shaderName) == 0) {
		shaderMap.emplace(shaderName, new XGLShader(shaderName));
		shaderMap[shaderName]->Compile(shaderName);
	}

	if (shapes->count(shaderName) == 0) {
		shapes->emplace(shaderName, new XGLShapeList);
	}

	XGLShape *pShape = fn();
	XGLShader *shader = shaderMap[shaderName];

	// XGLBuffer::Load() also makes this assignment, but for
	// geometryless shapes, it won't get called.  Do it here
	// just to be safe.
	pShape->shader = shader;

	if (pShape->v.size() > 0) {
		pShape->Load(shader, pShape->v, pShape->idx);
		pShape->uniformLocations = shader->materialLocations;
	}
	return pShape;
}

XGLShape* XGL::CreateShape(std::string shName, XGLNewShapeLambda fn, int layer){
	return CreateShape(shapeLayers[layer], shName, fn);
}

void XGL::AddShape(std::string shName, XGLNewShapeLambda fn, int layer){
	XGLShape *pShape = CreateShape(shName, fn, layer);

	(*shapeLayers[layer])[pShape->shader->Name()]->push_back(pShape);

	AddChild(pShape);
}

void XGL::AddGuiShape(std::string shName, XGLNewShapeLambda fn){
	XGLShape *pShape = CreateShape(&guiShapes, shName, fn);

	guiShapes[pShape->shader->Name()]->push_back(pShape);

	AddChild(pShape);

	if (guiManager == NULL)
		guiManager = (XGLGuiManager *)pShape;
	else
		guiManager->AddChild(pShape);
}

void XGL::IterateShapesMap(){
	for (auto shapes : shapeLayers) {
		for (auto perShader : *shapes) {
			XGLShader *shader = shaderMap[perShader.first];
			xprintf("XGL::IterateShapesMap(): '%s', shader->shader: %d\n", Name().c_str(), shader->programId);

			for (auto shape : *(perShader.second))
				xprintf("   shape->b: vbo:%d, program:%d\n", shape->vbo, shape->shader->programId);
		}
	}
}

bool XGL::GuiResolveMouseEvent(XGLShape *shape, int x, int y, int flags) {
	XObjectChildren guiChildren = shape->Children();
	XObjectChildren::reverse_iterator rit;
	bool handledByChild = false;
	glm::vec4 ul, lr, mc;

	if (mouseCaptured != NULL) {
		if (dynamic_cast<XGLGuiCanvas *>(mouseCaptured)) {
			XGLGuiCanvas *gc;
			glm::mat4 tmpModel = glm::mat4();

			// accumulate all ancestoral transformations
			for (gc = (XGLGuiCanvas *)mouseCaptured; gc->Parent() != nullptr; gc = (XGLGuiCanvas *)gc->Parent())
				tmpModel *= gc->model;

			// reset to captured XGLGuiCanvas
			gc = (XGLGuiCanvas *)mouseCaptured;

			// convert to window-relative coordinates
			mc = glm::inverse(tmpModel) * glm::vec4(x, y, 1, 1);
			handledByChild =  gc->MouseEvent(mc.x, mc.y, flags);
		}
	}
	else {
		// in case siblings are stacked on top of each other, going backwards
		// will return the top-most, as desired.  Otherwise it doesn't matter
		// what order we iterate in, ie: going backward does what's desired
		// with no negative consequences.
		for (rit = guiChildren.rbegin(); rit != guiChildren.rend(); rit++) {
			XGLShape *shape = (XGLShape *)*rit;

			if (dynamic_cast<XGLGuiCanvas *>(shape)) {
				XGLGuiCanvas *gc = (XGLGuiCanvas *)shape;
				ul = gc->model * glm::vec4(0.0, 0.0, 0.0, 1.0);
				lr = gc->model * glm::vec4(gc->width, gc->height, 0.0, 1.0);

				if ((x >= ul.x && x <= lr.x) && (y >= ul.y && y <= lr.y)){
					// convert to window-relative coordinates
					mc = glm::inverse(gc->model) * glm::vec4(x, y, 1.0, 1.0);

					// recurse into child stack (if there is one)
					if (gc->Children().size() > 0)
						handledByChild = GuiResolveMouseEvent(gc, (int)(mc.x), (int)(mc.y), flags);

					if (!handledByChild) {
						gc->SetHasMouse(true);
						handledByChild = gc->MouseEvent(mc.x, mc.y, flags);
						if (handledByChild)
							break;
					}
				}
				else {
					gc->SetHasMouse(false);
				}
			}
		}
	}

	return handledByChild;
}

void XGL::GetPreferredWindowSize(int *w, int *h) {
	*w = preferredWidth;
	*h = preferredHeight;
}

#define QUERY_GLCONTEXT( x, v ) { glGetIntegerv((x),&(v)); GL_CHECK("glGetIntegerv() failed"); xprintf("%s: %d\n", #x,(v)); }

void XGL::QueryContext() {
	GLint value = 0;

//	QUERY_GLCONTEXT(GL_MAX_UNIFORM_BUFFER_BINDINGS, value);
//	QUERY_GLCONTEXT(GL_MAX_UNIFORM_BLOCK_SIZE, value);
//	QUERY_GLCONTEXT(GL_MAX_VERTEX_UNIFORM_BLOCKS, value);
//	QUERY_GLCONTEXT(GL_MAX_FRAGMENT_UNIFORM_BLOCKS, value);
//	QUERY_GLCONTEXT(GL_MAX_GEOMETRY_UNIFORM_BLOCKS, value);
//	QUERY_GLCONTEXT(GL_UNIFORM_BUFFER_OFFSET_ALIGNMENT, value);
}

void  XGL::AddThread(std::shared_ptr<XThread> t) {
	threadPool.push_back(t);
}
