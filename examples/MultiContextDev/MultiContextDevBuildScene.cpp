/**************************************************************
** MultiContextDevBuildScene.cpp
**
** To support asynchronous image upload for multimedia apps,
** multiple OpenGL contexts are required.  Up until now
** XGL has only had the main OpenGL context.  XGLContext
** provides integration of OpenGL context management for
** multi-threaded OpenGL goodness.
**
** Start with a background thread / GL context that repeatedly 
** copies alternating frames to an oversized PBO to see what 
** opengl debug output has to say about buffer usage.
**
** Use XTimer to measure transfer to PBO and and transfer from
** PBO (via glTexSubImage2D())
**************************************************************/
#include "ExampleXGL.h"

static const int numFrames = 4;
#define INDEX(x) ( (x) % numFrames)

class XGLContext : public XThread {
public:
	XGLContext(ExampleXGL* pxgl, int w, int h, int c) : pXgl(pxgl), width(w), height(h), channels(c), XThread("XGLContextThread") {
		glfwSetErrorCallback(ErrorFunc);

		// need 2nd OpenGL context, which GLFW creates with glfwCreateWindow()
		// hint to GLFW that the window is not visible, and make it small to save memory
		glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);
		if ((mWindow = glfwCreateWindow(32, 32, "Offscreen", NULL, pXgl->window)) == nullptr)
			xprintf("Oops, glfwCreateWindow() failed\n");

		// make new OpenGL context current so we can set it up.
		glfwMakeContextCurrent(mWindow);

		// calculate the size of one image
		pboSize = width*height*channels;

		glGenBuffers(1, &pboId);
		GL_CHECK("glGenBuffers failed");

		xprintf("pboId: %d\n", pboId);

		// bind our PBO to UNPACK_BUFFER, allocate multiple frames, make the whole damn thing, persistently
		glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pboId);
		GL_CHECK("glBindBuffer() failed");

		glBufferStorage(GL_PIXEL_UNPACK_BUFFER, pboSize*numFrames, nullptr, pboFlags);
		GL_CHECK("glBufferStorage() failed");

		pboBuffer = (uint8_t*)glMapBufferRange(GL_PIXEL_UNPACK_BUFFER, 0, pboSize*numFrames, pboFlags);
		GL_CHECK("glMapBufferRange() failed");

		if (pboBuffer == nullptr)
			xprintf("Doh! glMapBufferRange() returned nullptr\n");

		// restore main OpenGL context.  (it can't be bound in 2 threads at once)
		glfwMakeContextCurrent(pXgl->window);

		black = new uint8_t[pboSize];
		memset(black, 0, pboSize);

		white = new uint8_t[pboSize];
		memset(white, 255, pboSize);

		// initialize fill and render fences
		for (int i = 0; i < numFrames; i++) {
			fillFences[i] = 0;
			renderFences[i] = 0;
		}
	}

	static void ErrorFunc(int code, const char *str) {
		xprintf("%s(): %d - %s\n", __FUNCTION__, code, str);
	}

	void Run() {
		glfwMakeContextCurrent(mWindow);

		while (IsRunning()) {
			int wIndex = INDEX(framesWritten);	// currently active frame for writing
			int offset = wIndex * pboSize;		// where it is in PBO

			// make sure "renderFence" is actually valid before waiting for it
			if (renderFences[wIndex])
				GLenum waitRet = glClientWaitSync(renderFences[wIndex], 0, 20000000);

			// time the copy to the PBO
			xtimer.SinceLast();
			if (framesWritten & 1)
				memcpy(pboBuffer + offset, black, pboSize);
			else
				memcpy(pboBuffer + offset, white, pboSize);
			double et = xtimer.SinceLast();

			// put a fence in so renedering thread can know if we're done with this frame
			fillFences[INDEX(framesWritten)] = glFenceSync(GL_SYNC_GPU_COMMANDS_COMPLETE, 0);
			GL_CHECK("glFenceSync() failed");

			// calculate transfer rate
			//double bytesPerSecond = pboSize / et;
			//xprintf("et: %0.8f, GB/s: %0.4f\n", et, bytesPerSecond / 1000000000.0);

			std::this_thread::sleep_for(std::chrono::duration<int, std::milli>(2));
			framesWritten++;
		}
	}

	GLFWwindow* mWindow;
	XTimer xtimer;
	ExampleXGL* pXgl;
	GLuint pboId;
	uint8_t* pboBuffer;
	GLbitfield pboFlags{ GL_MAP_WRITE_BIT | GL_MAP_PERSISTENT_BIT | GL_MAP_COHERENT_BIT };
	int width, height, channels;
	int pboSize;
	uint64_t framesWritten{ 0 };
	uint64_t framesRead{ 0 };
	uint8_t *black, *white;
	GLsync fillFences[numFrames];
	GLsync renderFences[numFrames];
};

// derive from XGLTexQuad a class that allows us to overide it's Draw() call so
// we can fiddle around with various asynchronous I/O transfer strategies.
class XGLContextImage : public XGLTexQuad {
public:
	XGLContextImage() : XGLTexQuad(960,540,4) {
		ta = texAttrs[0];
		pboSize = ta.width * ta.height * ta.channels;
		xprintf("imageSize: %d,%d,%d\n", ta.width, ta.height, ta.channels);

		glGenTextures(numFrames, textures);
		GL_CHECK("glGenTextures() didn't work");
	}

	void SetXGLContext(XGLContext *p) { pContext = p; }

	void Draw() {
		if (pContext->framesWritten < numFrames)
			return;

		// the texture mapping setup for this draw call has already been done
		// by the time we get here, so fiddling with the XGLContext generated
		// PBO data won't get hosed by the default XGLTexQuad::Draw() method.
		// So we can party on here with the async PBO here.
		// Make sure the XGLContext's PBO is bound to UNPACK_BUFFER
		glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pContext->pboId);
		GL_CHECK("glBindBuffer() didn't work.");

		int rIndex = INDEX(pContext->framesRead++);		// currently active frame for reading

		// wait till upload thread filling of buffer is complete.
		glWaitSync(pContext->fillFences[rIndex], 0, GL_TIMEOUT_IGNORED);
		GL_CHECK("glWaitSync() failed");

		// initiate transfer from PBO to texture.  Yes, this makes pixel transfer synchronous with rendering.
		xferTimer.SinceLast();
		glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, ta.width, ta.height, GL_RGB, GL_UNSIGNED_BYTE, (GLvoid *)(rIndex*pboSize));
		double et = xferTimer.SinceLast();
		GL_CHECK("glTexSubImage2D() failed");

		// signal upload thread we're done with this frame
		pContext->renderFences[rIndex] = glFenceSync(GL_SYNC_GPU_COMMANDS_COMPLETE, 0);
		GL_CHECK("glFenceSync() failed");

		// calculate transfer rate
		//double bytesPerSecond = pboSize / et;
		//xprintf("xt: %0.8f, GB/s: %0.4f\n", et, bytesPerSecond / 1000000000.0);

		// rely on base class for actual render of texture quad
		XGLTexQuad::Draw();
	}

	XGLContext *pContext{ nullptr };
	TextureAttributes ta;
	XTimer xferTimer;
	int pboSize;
	GLuint textures[numFrames];
};

void ExampleXGL::BuildScene() {
	XGLContextImage *shape;

	// get a new XGLContextImage() - currently hardcoded to 960 x 540 x 4 size
	AddShape("shaders/tex", [&shape](){ shape = new XGLContextImage(); return shape; });

	// get the data that tells us the dimensions of the texture buffer
	XGLShape::TextureAttributes ta = shape->texAttrs[0];

	// use dimensions of XGLContextImage when creating XGLContext
	XGLContext *ac = new XGLContext(this, ta.width, ta.height, ta.channels);

	// have the upright texture scaled up and made 16:9 aspect, and orbiting the origin
	// to highlight use of the callback function for animation of a shape.  Note that this function
	// runs once per frame BEFORE the shape's geomentry is rendered.  A lot can
	// be done here. Hint: scripting, physics(?)
	shape->SetAnimationFunction([shape](float clock) {
		glm::mat4 translate = glm::translate(glm::mat4(), glm::vec3(0, 0, 5.4f));
		glm::mat4 scale = glm::scale(glm::mat4(), glm::vec3(9.6f, 5.4f, 1.0f));
		glm::mat4 rotate = glm::rotate(glm::mat4(), glm::radians(90.0f), glm::vec3(1.0f, 0.0f, 0.0f));
		shape->model = translate * rotate * scale;
	});

	shape->SetXGLContext(ac);
	ac->Start();
}
