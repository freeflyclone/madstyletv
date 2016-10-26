#include "xgl.h"

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

// for GLUT runtimes
#ifdef USE_GLUT
XGL::XGL(int *argcp, char **argv) {
    instance.reset(this);

    glutInitDisplayMode( GLUT_3_2_CORE_PROFILE | GLUT_RGB | GLUT_DOUBLE | GLUT_DEPTH);
    glutInitWindowPosition(50,50);
    glutInitWindowSize(1280,720);
    glutInit(argcp, argv);
    glutCreateWindow("GLUT Window");

#ifndef OSX
	glewExperimental = GL_TRUE;

	if (glewInit() != GLEW_OK)
		throwXGLException("glewInit() failed");

	if (!GLEW_VERSION_2_1)
		throwXGLException("Requires OpenGL version 2.1 or greater");
#endif

    glutDisplayFunc(display);
    glutIdleFunc(idle);
    glutReshapeFunc(reshape);

	
    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LESS);
}
#endif

XGL::XGL() : XGLObject("XGL"), clock(0.0f), pb(NULL), fb(NULL) {
    xprintf("OpenGL version: %s\n", glGetString(GL_VERSION));
	glGetError();

	QueryContext();

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

	// for copying to shared memory buffer
	fb = new XGLSharedFBO();

//	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
//	glEnable(GL_BLEND);

//	glCullFace(GL_BACK);
//	glEnable(GL_CULL_FACE);
}

XGL::~XGL(){
    // iterate through all of the shapes, according to which shader they use
    XGLShapesMap::iterator perShader = shapes.begin();

    // iterate through all shapes associated with a particular shader 
    XGLShapeList::iterator perShape;

    // pointer to an XGLShader from the shaderMap
    XGLShader *shader;

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

void XGL::RenderScene() {
	camera.Animate();

	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	// set the projection,view,model,mvp matrices in the matrix UBO
	shaderMatrix.view = camera.GetViewMatrix();
	shaderMatrix.projection = projector.GetProjectionMatrix();

	glBufferData(GL_UNIFORM_BUFFER, sizeof(shaderMatrix), (GLvoid *)&shaderMatrix, GL_DYNAMIC_DRAW);
	GL_CHECK("glBufferData() failed");


    // iterate through all of the shaders...
	XGLShapesMap::iterator perShader;
    // ... then iterate through all shapes associated with the current shader 
    XGLShapeList::iterator eachShape;
    // pointer to an XGLShape from the perShape iterator
    XGLShape *shape;

	for (perShader = shapes.begin(); perShader != shapes.end(); perShader++) {
		XGLShapeList *shapeList = perShader->second;
		std::string name = perShader->first;
		XGLShader *shader = shaderMap[name];
		
		shader->Use();

		glUniform3fv(glGetUniformLocation(shader->programId, "cameraPosition"), 1, (GLfloat*)glm::value_ptr(camera.pos));
		GL_CHECK("glUniform3fv() failed");

		for (eachShape = shapeList->begin(); eachShape != shapeList->end(); eachShape++) {
			shape = *eachShape;
			shape->Render(clock);
		}

        shader->UnUse();
    }
}

void XGL::Display(){
	PreRender();

	glBindBufferBase(GL_UNIFORM_BUFFER, 0, matrixUbo);
	GL_CHECK("glBindBuffer() failed");

	projector.Reshape();
	RenderScene();

	if (fb)
		fb->Render(projector.width, projector.height);

	if (pb)
		pb->Render();

	clock += 1.0f;
}

void XGL::PreRender() {
	// iterate through all of the shaders...
	XGLShapesMap::iterator perShader;
	XGLShapeList::iterator eachShape;
	XGLShape *shape;

	for (perShader = shapes.begin(); perShader != shapes.end(); perShader++) {
		XGLShapeList *shapeList = perShader->second;
		std::string name = perShader->first;
		XGLShader *shader = shaderMap[name];

		shader->Use();

		for (eachShape = shapeList->begin(); eachShape != shapeList->end(); eachShape++) {
			shape = *eachShape;
			if (shape->preRenderFunction)
				shape->preRenderFunction(shape, clock);
		}

		shader->UnUse();
	}
}


XGLShape* XGL::CreateShape(std::string shName, XGLNewShapeLambda fn){
	std::string shaderName = pathToAssets + "/" + shName;
	if (shaderMap.count(shaderName) == 0) {
		shaderMap.emplace(shaderName, new XGLShader(shaderName));
		shaderMap[shaderName]->Compile(shaderName);
	}

	if (shapes.count(shaderName) == 0) {
		shapes.emplace(shaderName, new XGLShapeList);
	}

	XGLShape *pShape = fn();
	XGLShader *shader = shaderMap[shaderName];

	if (pShape->v.size() > 0) {
		pShape->Load(shader, pShape->v, pShape->idx);
		pShape->uniformLocations = shader->materialLocations;
	}
	return pShape;
}

void XGL::AddShape(std::string shName, XGLNewShapeLambda fn){
	XGLShape *pShape = CreateShape(shName, fn);

	shapes[pShape->shader->Name()]->push_back(pShape);

	AddChild(pShape);
}

void XGL::IterateShapesMap(){
    // iterate through all of the shapes, according to which shader they use
    XGLShapesMap::iterator perShader = shapes.begin();

    // iterate through all shapes associated with a particular shader 
    XGLShapeList::iterator perShape;

    // pointer to an XGLShader from the shaderMap
    XGLShader *shader;

    for (perShader = shapes.begin(); perShader != shapes.end(); perShader++) {
        std::string name = perShader->first;
        shader = shaderMap[name];
		xprintf("XGL::IterateShapesMap(): '%s', shader->shader: %d\n", name.c_str(), shader->programId);

        for (perShape = perShader->second->begin(); perShape != perShader->second->end(); perShape++) {
            XGLShape *shape = *perShape;
			xprintf("   shape->b: vao:%d, vbo:%d, program:%d\n", shape->vao, shape->vbo, shape->shader->programId);
        }
    }
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

#ifdef USE_GLUT
void XGL::display() {
	std::shared_ptr<XGL> px = getInstance();
	px->Display();
	glutSwapBuffers();
}
void XGL::reshape(int w, int h) {
	std::shared_ptr<XGL> px = getInstance();
	px->projector.Reshape(w,h);
}
void XGL::idle() {
	std::shared_ptr<XGL> px = getInstance();
	px->Idle();
	glutPostRedisplay();
}
#endif
