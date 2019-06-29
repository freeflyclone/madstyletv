#ifndef XGLCONTEXTIMAGE_H
#define XGLCONTEXTIMAGE_H

#include "ExampleXGL.h"
#include "xglpixelformat.h"

#define INDEX(x) ((x) % numFrames)

// derive from XGLTexQuad a class. That allows us to overide it's Draw() call so
// we can fiddle around with various asynchronous I/O transfer strategies.
class XGLContextImage : public XGLTexQuad , public XThread, public SteppedTimer {
public:
	XGLContextImage(ExampleXGL* pxgl, int w, int h) :
		width(w), height(h),
		XGLTexQuad(),
		mMainContextWindow(pxgl->window),
		XThread("XGLContextFpsThread")
	{
		// get pixel layout we expect from FFmpeg for GoPro footage
		ppfd = new XGLPixelFormatDescriptor(AV_PIX_FMT_YUVJ420P);

		// calculate the size of one image
		chromaWidth = width >> ppfd->shiftRightW;
		chromaHeight = height >> ppfd->shiftRightH;

		AddTexture(width, height, 1);
		AddTexture(chromaWidth, chromaHeight, 1);
		AddTexture(chromaWidth, chromaHeight, 1);

		glfwSetErrorCallback(ErrorFunc);

		// Need 2nd OpenGL context, which GLFW creates with glfwCreateWindow().
		// Hint to GLFW that the window is not visible, and make it small to save memory
		glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);
		if ((mWindow = glfwCreateWindow(32, 32, "Offscreen", NULL, mMainContextWindow)) == nullptr)
			xprintf("Oops, glfwCreateWindow() failed\n");

		// Trying out: 3rd OpenGL context for accurate FPS throttling... it's a WAG right now.
		glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);
		if ((mFpsWindow = glfwCreateWindow(32, 32, "Offscreen", NULL, mMainContextWindow)) == nullptr)
			xprintf("Oops, glfwCreateWindow() failed\n");

		// restore main OpenGL context.  (new context can't be bound in 2 threads at once)
		glfwMakeContextCurrent(mMainContextWindow);

		// ensure a texture unit, 0th is safest
		glActiveTexture(GL_TEXTURE0);

		// get a texture buffer for each of "numFrames" * "numPlanes"
		GLuint texId[numFrames*numPlanes];
		glGenTextures(numFrames*numPlanes, texId);

		for (int i = 0; i < numFrames; i++) {
			for (int j = 0; j < numPlanes; j++) {
				int w = (j == 0) ? width : chromaWidth;
				int h = (j == 0) ? height : chromaHeight;
				int tIdx = i * numPlanes + j;

				glActiveTexture(GL_TEXTURE0 + j);
				glBindTexture(GL_TEXTURE_2D, texId[tIdx]);
				glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
				glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
				glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
				glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
				glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
				glTexImage2D(GL_TEXTURE_2D, 0, GL_RED, w, h, 0, GL_RED, GL_UNSIGNED_BYTE, 0);

				bgTexIds.push_back(texId[tIdx]);
				bgTexAttrs.push_back({ w, h, 1 });
				bgNumTextures++;
			}
		}
		GL_CHECK("XGLContextImage::XGLContextImage(): something went wrong with texture allocation");

		// initialize fill and render fences
		for (int i = 0; i < numFrames; i++) {
			fillFences[i] = 0;
			renderFences[i] = 0;
		}
	}

	void Run() {
		while (IsRunning()) {
			GameTime gameTime;
			while (!TryAdvance(gameTime))
				std::this_thread::sleep_for(std::chrono::duration<int, std::milli>(1));
			freeFrames.notify();
		}
	}

	static void ErrorFunc(int code, const char *str) {
		xprintf("%s(): %d - %s\n", __FUNCTION__, code, str);
	}

	void MakeContextCurrent() { 
		glfwMakeContextCurrent(mWindow);
	}

	void UploadToTexture(uint8_t* y, uint8_t* u, uint8_t* v) {
		//xprintf("%s(): %0.5f,  y: %p, u: %p, v:%p\n", __FUNCTION__, xtUpload.SinceLast(), y, u, v);

		int wIndex = NextFree();	// currently active frame for writing

		// make sure "renderFence" is actually valid before waiting for it w/200ms timeout.
		if (renderFences[wIndex])
			glClientWaitSync(renderFences[wIndex], 0, 20000000);

		// Initiate DMA transfers to Y,U and V textures individually.
		// (it doesn't appear necessary to change texture units here)
		glBindTexture(GL_TEXTURE_2D, bgTexIds[wIndex*numPlanes]);
		glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height, GL_RED, GL_UNSIGNED_BYTE, (GLvoid *)y);
		glBindTexture(GL_TEXTURE_2D, bgTexIds[wIndex*numPlanes + 1]);
		glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, chromaWidth, chromaHeight, GL_RED, GL_UNSIGNED_BYTE, (GLvoid *)u);
		glBindTexture(GL_TEXTURE_2D, bgTexIds[wIndex*numPlanes + 2]);
		glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, chromaWidth, chromaHeight, GL_RED, GL_UNSIGNED_BYTE, (GLvoid *)v);
		GL_CHECK("glTexSubImage() failed");

		// put a fence in so renedering context can know if we're done with this frame
		fillFences[wIndex] = glFenceSync(GL_SYNC_GPU_COMMANDS_COMPLETE, 0);
		GL_CHECK("glFenceSync() failed");

		usedFrames.notify();
	}

	void Draw() {
		// wait until decoder has actually emitted it's 1st frame
		if (framesWritten < 1)
			return;

		// coded to return the MRU index, for multiple frames if need be.
		// So if preferredSwapInterval is set to 0, OpenGL goes full throttle,
		// vastly exceeding the monitors ability to keep sync.  But... if you're
		// desktop windowing system has a compositor, and these days most do, then
		// full throttle OpenGL doesn't tear. It aint' pretty, but it works.
		int rIndex = NextUsed();

		// proper set up of texUnit via Uniform is necessary, for reasons.  (can't hard code in shader?)
		// "yuv" shader expects Y,U,V in texture unit 0, 1 and 2 respectively.
		glProgramUniform1i(shader->programId, glGetUniformLocation(shader->programId, "texUnit0"), 0);
		glProgramUniform1i(shader->programId, glGetUniformLocation(shader->programId, "texUnit1"), 1);
		glProgramUniform1i(shader->programId, glGetUniformLocation(shader->programId, "texUnit2"), 2);

		// wait till upload thread filling of buffer(s) is complete.
		glWaitSync(fillFences[rIndex], 0, GL_TIMEOUT_IGNORED);
		GL_CHECK("glWaitSync() failed");

		// True asynchronous texture upload... just bind the relevant texture(s)
		// that were uploaded by the upload thread
		// Y
		glActiveTexture(GL_TEXTURE0);
		glBindTexture(GL_TEXTURE_2D, bgTexIds[rIndex*numPlanes]);

		// U
		glActiveTexture(GL_TEXTURE1);
		glBindTexture(GL_TEXTURE_2D, bgTexIds[rIndex*numPlanes + 1]);

		// V
		glActiveTexture(GL_TEXTURE2);
		glBindTexture(GL_TEXTURE_2D, bgTexIds[rIndex*numPlanes + 2]);

		GL_CHECK("glBindTexture() failed");

		// signal upload thread we're done with this frame
		renderFences[rIndex] = glFenceSync(GL_SYNC_GPU_COMMANDS_COMPLETE, 0);
		GL_CHECK("glFenceSync() failed");

		// rely on base class for actual render of texture quad
		XGLTexQuad::Draw();
	}

	~XGLContextImage() {
		xprintf("%s()\n", __FUNCTION__);
		if (IsRunning())
			WaitForStop();
	}

	int NextFree() {
		if(!freeFrames.wait_for(100))
			return INDEX(framesWritten);
		return INDEX(framesWritten++);
	}

	int NextUsed() {
		if (!usedFrames.wait_for(1))
			return INDEX(framesRead);

		// leave at least on frame as "cushion" otherwise we end up waiting on the decoder
		// instead of the fps timer.
		if (framesWritten-framesRead <= 1)
			return INDEX(framesRead);

		return INDEX(framesRead++);
	}

	static const int numPlanes{ 3 };
	static const int numFrames{ 2 };

	// alternate OpenGL context things
	GLFWwindow* mWindow;
	GLFWwindow* mFpsWindow;
	GLFWwindow* mMainContextWindow;

	// image geometry luma & chroma
	int width{ 0 }, height{ 0 };
	int chromaWidth{ 0 }, chromaHeight{ 0 };
	XGLPixelFormatDescriptor* ppfd{ nullptr };

	// "circular" image buffer management
	uint64_t framesWritten{ 0 };
	uint64_t framesRead{ 0 };

	XSemaphore usedFrames{ 0 };
	XSemaphore freeFrames{ numFrames };

	// GL syncronization stuff
	GLsync fillFences[numFrames];
	GLsync renderFences[numFrames];
	GLsync fpsFence;

	// texture management stuff
	std::vector<GLuint> bgTexIds;
	XGLBuffer::TextureAttributesList bgTexAttrs;
	GLuint bgNumTextures{ 0 };

	XTimer xtUpload;
};




#endif
