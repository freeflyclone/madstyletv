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
** copies alternating frames to a PBO to see what opengl debug
** output has to say about buffer usage.
**
** Use XTimer to measure transfer to PBO and and transfer from
** PBO (via glTexSubImage2D())
**************************************************************/
#include "ExampleXGL.h"

static const int numFrames = 4;
static const int numFramesMask = (numFrames - 1);
#define INDEX(x) ( (x) % numFrames)

class XGLContext : public XThread {
public:
	XGLContext(ExampleXGL* pxgl, int w, int h, int c) : pXgl(pxgl), width(w), height(h), channels(c), XThread("XGLContextThread") {
		glfwSetErrorCallback(ErrorFunc);
		glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);

		if ((mWindow = glfwCreateWindow(32, 32, "Offscreen", NULL, pXgl->window)) == nullptr)
			xprintf("Oops, glfwCreateWindow() failed\n");

		glfwMakeContextCurrent(mWindow);

		pboSize = width*height*channels;

		glGenBuffers(1, &pboId);
		GL_CHECK("glGenBuffers failed");

		xprintf("pboId: %d\n", pboId);

		glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pboId);
		GL_CHECK("glBindBuffer() failed");

		glBufferStorage(GL_PIXEL_UNPACK_BUFFER, pboSize*numFrames, nullptr, pboFlags);
		GL_CHECK("glBufferStorage() failed");

		pboBuffer = (uint8_t*)glMapBufferRange(GL_PIXEL_UNPACK_BUFFER, 0, pboSize*numFrames, pboFlags);
		GL_CHECK("glMapBufferRange() failed");

		if (pboBuffer == nullptr) {
			xprintf("Doh! glMapBufferRange() returned nullptr\n");
		}

		glfwMakeContextCurrent(pXgl->window);

		black = new uint8_t[pboSize];
		memset(black, 0, pboSize);

		white = new uint8_t[pboSize];
		memset(white, 255, pboSize);
	}

	static void ErrorFunc(int code, const char *str) {
		xprintf("%s(): %d - %s\n", __FUNCTION__, code, str);
	}

	void Run() {
		glfwMakeContextCurrent(mWindow);

		while (IsRunning()) {
			xtimer.SinceLast();
			int offset = INDEX(framesWritten) * pboSize;

			if (framesWritten & 1)
				memcpy(pboBuffer + offset, black, pboSize);
			else
				memcpy(pboBuffer + offset, white, pboSize);

			fences[INDEX(framesWritten)] = glFenceSync(GL_SYNC_GPU_COMMANDS_COMPLETE, 0);
			GL_CHECK("glFenceSync() failed");

			double et = xtimer.SinceLast();
			double bytesPerSecond = pboSize / et;
			xprintf("et: %0.8f, GB/s: %0.4f\n", et, bytesPerSecond / 1000000000.0);

			std::this_thread::sleep_for(std::chrono::duration<int, std::milli>(10));
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
	GLsync fences[numFrames];
};

class XGLContextImage : public XGLTexQuad {
public:
	XGLContextImage() : XGLTexQuad(960,540,4) {
		ta = texAttrs[0];
		pboSize = ta.width * ta.height * ta.channels;
		xprintf("imageSize: %d,%d,%d\n", ta.width, ta.height, ta.channels);

		glGenTextures(numFrames, textures);
		GL_CHECK("glGenTextures() didn't work");
	}

	void SetContext(XGLContext *p) { pContext = p; }

	void Draw() {
		if (pContext->framesWritten < numFrames)
			return;

		// the texture mapping setup for this draw call has already been done
		// by the time we get here, so fiddling with the XGLContext generated
		// PBO data won't get hosed by the default XGLTexQuad::Draw() method.
		// So we can party on here with the async PBO here.

		glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pContext->pboId);
		GL_CHECK("glBindBuffer() didn't work.");

		xferTimer.SinceLast();
		
		int index = INDEX(pContext->framesRead++);

		glWaitSync(pContext->fences[index], 0, GL_TIMEOUT_IGNORED);

		glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, ta.width, ta.height, GL_RGB, GL_UNSIGNED_BYTE, (GLvoid *)(index*pboSize));

		double et = xferTimer.SinceLast();
		double bytesPerSecond = pboSize / et;
		xprintf("xt: %0.8f, GB/s: %0.4f\n", et, bytesPerSecond / 1000000000.0);

		GL_CHECK("glTexSubImage2D() failed");

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

	AddShape("shaders/tex", [&shape](){ shape = new XGLContextImage(); return shape; });

	XGLShape::TextureAttributes ta = shape->texAttrs[0];

	XGLContext *ac = new XGLContext(this, ta.width, ta.height, ta.channels);

	// have the upright texture scaled up and made 16:9 aspect, and orbiting the origin
	// to highlight use of the callback function for animation of a shape.  Note that this function
	// runs once per frame BEFORE the shape's geomentry is rendered.  A lot can
	// be done here. Hint: scripting, physics(?)
	shape->SetAnimationFunction([shape](float clock) {
		float sinFunc = sin(clock / 40.0f) * 10.0f;
		float cosFunc = cos(clock / 40.0f) * 10.0f;
		glm::mat4 scale = glm::scale(glm::mat4(), glm::vec3(9.6f, 5.4f, 1.0f));
		glm::mat4 translate = glm::translate(glm::mat4(), glm::vec3(cosFunc, sinFunc, 5.4f));
		glm::mat4 rotate = glm::rotate(glm::mat4(), glm::radians(90.0f), glm::vec3(1.0f, 0.0f, 0.0f));
		shape->model = translate * rotate * scale;
	});

	shape->SetContext(ac);
	ac->Start();
}
