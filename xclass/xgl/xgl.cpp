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

XGL::XGL() : clock(0.0f), pb(NULL), fb(NULL), renderGui(false), guiManager(NULL), mouseCaptured(NULL) {
	SetName("XGL");
	xprintf("OpenGL version: %s\n", glGetString(GL_VERSION));
	glGetError();

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
	
	for (perShader = shapes.begin(); perShader != shapes.end(); perShader++) {
        std::string name = perShader->first;
        shader = shaderMap[name];

        for (perShape = perShader->second->begin(); perShape != perShader->second->end(); perShape++) {
            XGLShape *shape = *perShape;
            delete shape;
        }
        delete perShader->second;
        delete shader;
    }
}

void XGL::RenderScene(XGLShapesMap *shapes) {
	camera.Animate();

	// set the projection,view,orthoProjection matrices in the matrix UBO
	shaderMatrix.view = camera.GetViewMatrix();
	shaderMatrix.projection = projector.GetProjectionMatrix();
	shaderMatrix.orthoProjection = projector.GetOrthoMatrix();

	glBufferData(GL_UNIFORM_BUFFER, sizeof(shaderMatrix), (GLvoid *)&shaderMatrix, GL_DYNAMIC_DRAW);
	GL_CHECK("glBufferData() failed");

	for (auto perShader : *shapes) {
		XGLShader *shader = shaderMap[perShader.first];
		
		shader->Use();

		glUniform3fv(glGetUniformLocation(shader->programId, "cameraPosition"), 1, (GLfloat*)glm::value_ptr(camera.pos));
		GL_CHECK("glUniform3fv() failed");

		for (auto shape : *(perShader.second))
			shape->Render(clock);

        shader->UnUse();
    }
}

void XGL::Display(){
	PreRender();

	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	GL_CHECK("glClear() failed");

	glBindBufferBase(GL_UNIFORM_BUFFER, 0, matrixUbo);
	GL_CHECK("glBindBuffer() failed");

	// render the world
	RenderScene(&shapes);

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

	clock += 1.0f;
}

void XGL::PreRender() {
	for (auto perShader : shapes) {
		XGLShader *shader = shaderMap[perShader.first];
		
		shader->Use();

		for (auto shape : *(perShader.second))
			if (shape->preRenderFunction)
				shape->preRenderFunction(clock);

		shader->UnUse();
	}
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

XGLShape* XGL::CreateShape(std::string shName, XGLNewShapeLambda fn){
	return CreateShape(&shapes, shName, fn);
}

void XGL::AddShape(std::string shName, XGLNewShapeLambda fn){
	XGLShape *pShape = CreateShape(shName, fn);

	shapes[pShape->shader->Name()]->push_back(pShape);

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
	for (auto perShader : shapes) {
        XGLShader *shader = shaderMap[perShader.first];
		xprintf("XGL::IterateShapesMap(): '%s', shader->shader: %d\n", name.c_str(), shader->programId);

		for (auto shape : *(perShader.second))
			xprintf("   shape->b: vao:%d, vbo:%d, program:%d\n", shape->vao, shape->vbo, shape->shader->programId);
    }
}

bool XGL::GuiResolveMouseEvent(XGLShape *shape, int x, int y, int flags) {
	XGLObjectChildren guiChildren = shape->Children();
	XGLObjectChildren::reverse_iterator rit;
	bool handledByChild = false;
	glm::vec4 ul, lr, mc;

	if (mouseCaptured != NULL) {
		if (dynamic_cast<XGLGuiCanvas *>(mouseCaptured)) {
			XGLGuiCanvas *gc = (XGLGuiCanvas *)mouseCaptured;
			// convert to window-relative coordinates
			mc = glm::inverse(gc->model) * glm::vec4(x, y, 1, 1);
			handledByChild = gc->MouseEvent(mc.x, mc.y, flags);
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

#define QUERY_GLCONTEXT( x, v ) { glGetIntegerv((x),&(v)); GL_CHECK("glGetIntegerv() failed"); xprintf("%s: %d\n", #x,(v)); }

void XGL::QueryContext() {
	GLint value = 0;

	QUERY_GLCONTEXT(GL_MAX_UNIFORM_BUFFER_BINDINGS, value);
	QUERY_GLCONTEXT(GL_MAX_UNIFORM_BLOCK_SIZE, value);
	QUERY_GLCONTEXT(GL_MAX_VERTEX_UNIFORM_BLOCKS, value);
	QUERY_GLCONTEXT(GL_MAX_FRAGMENT_UNIFORM_BLOCKS, value);
	QUERY_GLCONTEXT(GL_MAX_GEOMETRY_UNIFORM_BLOCKS, value);
	QUERY_GLCONTEXT(GL_UNIFORM_BUFFER_OFFSET_ALIGNMENT, value);
}
