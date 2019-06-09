#ifndef XGLCONTEXTIMAGE_H
#define XGLCONTEXTIMAGE_H

#include "ExampleXGL.h"
#include "xglpixelformat.h"

static const int numPlanes = 3;
static const int numFrames = 4;

#define INDEX(x) ((x) % numFrames)

extern uint8_t* pGlobalPboBuffer;

// derive from XGLTexQuad a class. That allows us to overide it's Draw() call so
// we can fiddle around with various asynchronous I/O transfer strategies.
//
// Also derive from XThread for async buffer upload in alternate OpenGL context.
// (hence the name)
class XGLContextImage : public XGLTexQuad, public XThread {
public:
	// convenience ptrs to y,u,v buffers for each frame
	typedef struct { uint8_t *y, *u, *v; } YUV;

	XGLContextImage(ExampleXGL* pxgl, int w, int h, int c) :
		pXgl(pxgl),
		width(w), height(h), components(c),
		freeBuffs(numFrames),
		usedBuffs(0),
		XGLTexQuad(w, h, c),
		XThread("XGLContextImageThread") 
	{
		xprintf("%s()\n", __FUNCTION__);

		// get pixel layout we expect from FFmpeg for GoPro footage
		ppfd = new XGLPixelFormatDescriptor(AV_PIX_FMT_YUVJ420P);

		// calculate the size of one image
		chromaWidth = width >> ppfd->shiftRightW;
		chromaHeight = height >> ppfd->shiftRightH;

		AddTexture(chromaWidth, chromaHeight, 1);
		AddTexture(chromaWidth, chromaHeight, 1);

		glfwSetErrorCallback(ErrorFunc);

		// need 2nd OpenGL context, which GLFW creates with glfwCreateWindow()
		// hint to GLFW that the window is not visible, and make it small to save memory
		glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);
		if ((mWindow = glfwCreateWindow(32, 32, "Offscreen", NULL, pXgl->window)) == nullptr)
			xprintf("Oops, glfwCreateWindow() failed\n");

		// make new OpenGL context current so we can set it up.
		glfwMakeContextCurrent(mWindow);

		ySize = width*height*ppfd->depths[0];
		uvSize = chromaWidth*chromaHeight*ppfd->depths[1];
		pboSize = ySize + (uvSize * 2);

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
		pGlobalPboBuffer = pboBuffer = (uint8_t*)glMapBufferRange(GL_PIXEL_UNPACK_BUFFER, 0, pboSize*numFrames, pboFlags);
		GL_CHECK("glMapBufferRange() failed");

		if (pboBuffer == nullptr)
			throwXGLException("Doh! glMapBufferRange() returned nullptr\n");

		// restore main OpenGL context.  (new context can't be bound in 2 threads at once)
		glfwMakeContextCurrent(pXgl->window);

		// ensure a texture unit, 0th is safest
		glActiveTexture(GL_TEXTURE0);
		// get a texture buffer for each of "numFrames" * "numPlanes"
		GLuint texId[numFrames*numPlanes];
		glGenTextures(numFrames*numPlanes, texId);

		for (int i = 0; i < numFrames; i++) {
			// for easier access from clients
			{
				uint8_t *y, *u, *v;
				y = pboBuffer + (pboSize*i);
				u = y + ySize;
				v = u + uvSize;

				yuv[i] = { y, u, v };
			}

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
				bgTexAttrs.push_back({ w, h, components });
				bgNumTextures++;
			}
		}
		GL_CHECK("XGLContextImage::XGLContextImage(): something went wrong with texture allocation");

		// init a black image buffer (YUV420)
		black = new uint8_t[pboSize];
		memset(black, 0, ySize);
		memset(black + ySize, 127, uvSize);
		memset(black + ySize + uvSize, 127, uvSize);

		// init a white image buffer (YUV420)
		white = new uint8_t[pboSize];
		memset(white, 127, ySize);
		memset(white + ySize, 255, uvSize);
		memset(white + ySize + uvSize, 0, uvSize);

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
			/*
			if (framesWritten & 1) {
				memcpy(pboBuffer + offset, black, pboSize);
			}
			else {
				memcpy(pboBuffer + offset, white, pboSize);
			}
			*/

			// Initiate DMA transfers to Y,U and V textures individually.
			// (it doesn't appear necessary to change texture units here)
			glBindTexture(GL_TEXTURE_2D, bgTexIds[wIndex*numPlanes]);
			glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height, GL_RED, GL_UNSIGNED_BYTE, (GLvoid *)(wIndex*pboSize));
			glBindTexture(GL_TEXTURE_2D, bgTexIds[wIndex*numPlanes + 1]);
			glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, chromaWidth, chromaHeight, GL_RED, GL_UNSIGNED_BYTE, (GLvoid *)(wIndex*pboSize + ySize));
			glBindTexture(GL_TEXTURE_2D, bgTexIds[wIndex*numPlanes + 2]);
			glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, chromaWidth, chromaHeight, GL_RED, GL_UNSIGNED_BYTE, (GLvoid *)(wIndex*pboSize + ySize + uvSize));
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
		WaitForStop();
	}

	YUV* NextFree() {
		if (!freeBuffs.wait_for(100))
			return nullptr;

		return &yuv[(freeIdx++)&(numFrames - 1)];
	};

	// alternate OpenGL context things
	GLFWwindow* mWindow;
	ExampleXGL* pXgl;

	// PBO stuff
	int pboSize;
	GLuint pboId;
	uint8_t* pboBuffer;
	GLbitfield pboFlags{ GL_MAP_WRITE_BIT | GL_MAP_PERSISTENT_BIT | GL_MAP_COHERENT_BIT };

	// image geometry luma & chroma
	int width, height, components;
	int chromaWidth, chromaHeight;
	XGLPixelFormatDescriptor* ppfd{ nullptr };
	int ySize{ 0 }, uvSize{ 0 };

	// "circular" image buffer management
	uint64_t framesWritten{ 0 };
	uint64_t framesRead{ 0 };
	uint64_t freeIdx{ 0 };
	uint64_t usedIdx{ 0 };

	// high contrast dummy frames
	uint8_t *black, *white;

	// GL syncronization stuff
	GLsync fillFences[numFrames];
	GLsync renderFences[numFrames];

	// texture management stuff
	YUV yuv[numFrames];

	std::vector<GLuint> bgTexIds;
	XGLBuffer::TextureAttributesList bgTexAttrs;
	GLuint bgNumTextures{ 0 };

	XSemaphore freeBuffs, usedBuffs;
};




#endif
