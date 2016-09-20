#include "xgl.h"

std::string currentWorkingDir;
std::string pathToAssets;

void CheckError(const char *file, int line, std::string what){
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

void CheckStatus(const char *file, int line) {
	GLenum err = glGetError();
	if (err) {
		const GLubyte *errString = gluErrorString(err);
		xprintf("%s:%d - %s\n", file, line, errString);
	}
}

static std::shared_ptr<XGL> instance(NULL);
std::shared_ptr<XGL> XGL::getInstance() {
	return instance;
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

XGL::XGL() : XGLObject("XGL"), clock(0.0f) {
    instance.reset(this);
    
    xprintf("OpenGL version: %s\n", glGetString(GL_VERSION));
	glGetError();

	QueryContext();

	// utilize UBO for lighting parameters.
	{
		GLuint ubo;
		glGenBuffers(1, &ubo);
		GL_CHECK("glGenBuffers() failed");

		glBindBuffer(GL_UNIFORM_BUFFER, ubo);
		GL_CHECK("glBindBuffer() failed");

		// for now, create one point light, for diffuse lighting shader development
		XGLLight light = { { 10,6,16 },{ 1,1,1 }, 0.001f, 0.001f };
		lights.push_back(light);

		glBufferData(GL_UNIFORM_BUFFER, sizeof(light), &light, GL_DYNAMIC_DRAW);
		GL_CHECK("glBufferData() failed");

		glBindBufferBase(GL_UNIFORM_BUFFER, 1, ubo);
		GL_CHECK("glBindBufferBase() failed");
	}

	// utilize UBO for Matrix data
	{
		GLuint ubo;
		glGenBuffers(1, &ubo);
		GL_CHECK("glGenBuffers() failed");

		glBindBuffer(GL_UNIFORM_BUFFER, ubo);
		GL_CHECK("glBindBuffer() failed");

		glBufferData(GL_UNIFORM_BUFFER, sizeof(shaderMatrix), &shaderMatrix, GL_DYNAMIC_DRAW);
		GL_CHECK("glBufferData() failed");

		glBindBufferBase(GL_UNIFORM_BUFFER, 0, ubo);
		GL_CHECK("glBindBufferBase() failed");
	}

	{
		XGLLight l = lights.back();
		XGLColor c = l.diffuse;
		XGLVertex p = l.position;
		xprintf("Light color: %0.2f, %0.2f, %0.2f\n", c.r, c.g, c.b);
		xprintf("Light position: %0.2f, %0.2f, %0.2f\n", p.x, p.y, p.z);
	}

    font = std::make_shared<XGLFont>();

    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);

	glDepthFunc(GL_LESS);
	glEnable(GL_DEPTH_TEST);

//	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
//	glEnable(GL_BLEND);

//	glCullFace(GL_BACK);
//	glDisable(GL_CULL_FACE);
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

void XGL::Display() {
	camera.Animate();

	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	// set the projection,view,model,mvp matrices in the matrix UBO
	// this should probably be done else where, once startup-order issues are resolved
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
        shaderMap[name]->Use();
		XGLShader *shader = shaderMap[name];

		glUniform3fv(glGetUniformLocation(shader->Id(), "cameraPosition"), 1, (GLfloat*)glm::value_ptr(camera.pos));
		GL_CHECK("glUniform3fv() failed");

		for (eachShape = shapeList->begin(); eachShape != shapeList->end(); eachShape++) {
			shape = *eachShape;
			shape->Render(clock);
		}

        shaderMap[name]->UnUse();
    }

    clock += 1.0f;
}

void XGL::AddShape(std::string shName, XGLNewShapeLambda fn){
	std::string shaderName = pathToAssets + "/" + shName;
    if (shaderMap.count(shaderName) == 0) {
        shaderMap.emplace(shaderName, new XGLShader(shName));
        shaderMap[shaderName]->Compile(shaderName);
    }

    if (shapes.count(shaderName) == 0) {
        shapes.emplace(shaderName, new XGLShapeList);
    }

	XGLShader *shader = shaderMap[shaderName];
	shader->SetLights(&lights);
 
	XGLShape *pShape = fn();
	pShape->Load(shader, pShape->v, pShape->idx);
	AddChild(pShape);
    shapes[shaderName]->push_back(pShape);
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
		xprintf("XGL::IterateShapesMap(): '%s', shader->shader: %d\n", name.c_str(), shader->Id());

        for (perShape = perShader->second->begin(); perShape != perShader->second->end(); perShape++) {
            XGLShape *shape = *perShape;
			xprintf("   shape->b: vao:%d, vbo:%d, program:%d\n", shape->vao, shape->vbo, shape->program);
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
