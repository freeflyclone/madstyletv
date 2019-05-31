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
** copies alternating static frames to an oversized "circular"
** PBO that is mapped persistently, so it's always available 
** to CPU code.
**
** The upload thread simulates the output of a video decoding 
** thread from FFMpeg.
**
** The threads upload thread and main rendering thread use
** OpenGL GLsync objects to stay in sync, thus no tearing
** of the texture image is visible in the output.
**
** XTimer is used for precise timing measurements.
**************************************************************/
#include "ExampleXGL.h"

static const int numFrames = 4;
static const int vWidth = 1920;
static const int vHeight = 1080;
static const int nChannels = 1;

#define INDEX(x) ( (x) % numFrames)

// derive from XGLTexQuad a class. That allows us to overide it's Draw() call so
// we can fiddle around with various asynchronous I/O transfer strategies.
//
// Also derive from XThread for async buffer upload in alternate OpenGL context.
// (hence the name)
class XGLContextImage : public XGLTexQuad, public XThread {
public:
	XGLContextImage(ExampleXGL* pxgl, int w, int h, int c) : 
		pXgl(pxgl), 
		width(w), height(h), channels(c), 
		XGLTexQuad(vWidth, vHeight, nChannels), 
		XThread("XGLContextImageThread") 
	{
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

		xprintf("pboId: %d\n", pboId); // aid in identifying our PBO in GL debug output

		// Bind our PBO to UNPACK_BUFFER.
		glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pboId);
		GL_CHECK("glBindBuffer() failed");

		// Allocate multiple frames.
		glBufferStorage(GL_PIXEL_UNPACK_BUFFER, pboSize*numFrames, nullptr, pboFlags);
		GL_CHECK("glBufferStorage() failed");

		// Map the whole damn thing, persistently.
		pboBuffer = (uint8_t*)glMapBufferRange(GL_PIXEL_UNPACK_BUFFER, 0, pboSize*numFrames, pboFlags);
		GL_CHECK("glMapBufferRange() failed");

		if (pboBuffer == nullptr)
			throwXGLException("Doh! glMapBufferRange() returned nullptr\n");

		// restore main OpenGL context.  (new context can't be bound in 2 threads at once)
		glfwMakeContextCurrent(pXgl->window);

		// ensure a texture unit, 0th is safest
		glActiveTexture(GL_TEXTURE0);

		// get a texture buffer for each of "numFrames"
		GLuint texId[numFrames];
		glGenTextures(numFrames, texId);

		for (int i = 0; i < numFrames; i++) {
			glBindTexture(GL_TEXTURE_2D, texId[i]);
			glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
			glTexImage2D(GL_TEXTURE_2D, 0, GL_RED, width, height, 0, GL_RED, GL_UNSIGNED_BYTE, 0);

			texIds.push_back(texId[i]);
			texAttrs.push_back({vWidth, vHeight, nChannels});
			numTextures++;
		}
		GL_CHECK("XGLContextImage::XGLContextImage(): something went wrong with texture allocation");

		ta = texAttrs[0];
		xprintf("imageSize: %d,%d,%d\n", ta.width, ta.height, ta.channels);

		// init a black image buffer
		black = new uint8_t[pboSize];
		if (false)
		{
			uint32_t* bWords = (uint32_t*)black;
			for (int i = 0; i < pboSize / 4; i++)
				// layout: ABGR - little endian RGBA
				bWords[i] = 0xFF000000;
		}
		else
			memset(black, 0, pboSize);

		// init a white image buffer
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
		// first up, bind the new OpenGL context, for 2nd GPU command queue
		glfwMakeContextCurrent(mWindow);

		while (IsRunning()) {
			int wIndex = INDEX(framesWritten);	// currently active frame for writing
			int offset = wIndex * pboSize;		// where it is in PBO

			// make sure "renderFence" is actually valid before waiting for it w/200ms timeout.
			if (renderFences[wIndex])
				glClientWaitSync(renderFences[wIndex], 0, 20000000);

			// simulate what an ffmpeg decoder thread would do per frame
			if (framesWritten & 1)
				memcpy(pboBuffer + offset, black, pboSize);
			else
				memcpy(pboBuffer + offset, white, pboSize);

			glBindTexture(GL_TEXTURE_2D, texIds[wIndex]);
			GL_CHECK("glBindTexture() failed");

			// Initiate a DMA from the PBO to the texture memory in the new context
			glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height, GL_RED, GL_UNSIGNED_BYTE, (GLvoid *)(wIndex*pboSize));
			GL_CHECK("glTexSubImage() failed");

			// put a fence in so renedering context can know if we're done with this frame
			fillFences[INDEX(framesWritten)] = glFenceSync(GL_SYNC_GPU_COMMANDS_COMPLETE, 0);
			GL_CHECK("glFenceSync() failed");

			std::this_thread::sleep_for(std::chrono::duration<int, std::milli>(2));
			framesWritten++;
		}
	}

	void Draw() {
		if (framesWritten < numFrames)
			return;

		int rIndex = INDEX(framesRead++);		// currently active frame for reading

		// wait till upload thread filling of buffer is complete.
		glWaitSync(fillFences[rIndex], 0, GL_TIMEOUT_IGNORED);
		GL_CHECK("glWaitSync() failed");

		// True asynchronous texture upload... just bind the relevant texture
		// that was uploaded by the upload thread of XGLContext
		glBindTexture(GL_TEXTURE_2D, texIds[rIndex]);
		GL_CHECK("glBindTexture() failed");

		// signal upload thread we're done with this frame
		renderFences[rIndex] = glFenceSync(GL_SYNC_GPU_COMMANDS_COMPLETE, 0);
		GL_CHECK("glFenceSync() failed");

		// rely on base class for actual render of texture quad
		XGLTexQuad::Draw();
	}

	~XGLContextImage() {
		WaitForStop();
	}

	TextureAttributes ta;
	XTimer xferTimer;
	int pboSize;
	GLuint textures[numFrames];

	GLFWwindow* mWindow;
	//XTimer xtimer;
	ExampleXGL* pXgl;

	// PBO stuff
	GLuint pboId;
	uint8_t* pboBuffer;
	GLbitfield pboFlags{ GL_MAP_WRITE_BIT | GL_MAP_PERSISTENT_BIT | GL_MAP_COHERENT_BIT };
	int width, height, channels;

	// "circular" image buffer management
	uint64_t framesWritten{ 0 };
	uint64_t framesRead{ 0 };

	uint8_t *black, *white;

	// GL syncronization stuff
	GLsync fillFences[numFrames];
	GLsync renderFences[numFrames];

	// texture management stuff
	std::vector<GLuint> texIds;
	GLuint numTextures{ 0 };

};

void ExampleXGL::BuildScene() {
	XGLContextImage *shape;

	// get a new XGLContextImage()
	AddShape("shaders/yyy", [&shape,this](){ shape = new XGLContextImage(this, vWidth, vHeight, nChannels); return shape; });

	glm::mat4 translate = glm::translate(glm::mat4(), glm::vec3(0, 0, 5.4f));
	glm::mat4 scale = glm::scale(glm::mat4(), glm::vec3(9.6f, 5.4f, 1.0f));
	glm::mat4 rotate = glm::rotate(glm::mat4(), glm::radians(90.0f), glm::vec3(1.0f, 0.0f, 0.0f));
	shape->model = translate * rotate * scale;

	shape->Start();
}
