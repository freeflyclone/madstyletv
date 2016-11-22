#include "ExampleXGL.h"
#include <AntTweakBar.h>

class ATBShape : public XGLShape {
public:
	ATBShape(ExampleXGL *xgl) : flags(0), pxgl(xgl), speed(1.0), time(1.0) {
		xprintf("ATBShape::ATBShape()\n");
		TwInit(TW_OPENGL_CORE, NULL);
		TwBar *bar = TwNewBar("MadStyle TV");

		// Add 'speed' to 'bar': it is a modifable (RW) variable of type TW_TYPE_DOUBLE. Its key shortcuts are [s] and [S].
		TwAddVarRW(bar, "speed", TW_TYPE_DOUBLE, &speed,
			" label='Rot speed' min=0 max=2 step=0.01 keyIncr=s keyDecr=S help='Rotation speed (turns/second)' ");

		// Add 'wire' to 'bar': it is a modifable variable of type TW_TYPE_BOOL32 (32 bits boolean). Its key shortcut is [w].
		TwAddVarRW(bar, "wire", TW_TYPE_BOOL32, &wire,
			" label='Wireframe mode' key=w help='Toggle wireframe display mode.' ");

		// Add 'time' to 'bar': it is a read-only (RO) variable of type TW_TYPE_DOUBLE, with 1 precision digit
		TwAddVarRO(bar, "time", TW_TYPE_DOUBLE, &time, " label='Time' precision=1 help='Time (in seconds).' ");

		// Add 'bgColor' to 'bar': it is a modifable variable of type TW_TYPE_COLOR3F (3 floats color)
		TwAddVarRW(bar, "bgColor", TW_TYPE_COLOR3F, &bgColor, " label='Background color' ");

		// Add 'cubeColor' to 'bar': it is a modifable variable of type TW_TYPE_COLOR32 (32 bits color) with alpha
		TwAddVarRW(bar, "cubeColor", TW_TYPE_COLOR32, &cubeColor,
			" label='Cube color' alpha help='Color and transparency of the cube.' ");

		pxgl->projector.AddReshapeCallback(std::bind(&ATBShape::Reshape, this, _1, _2));
		pxgl->AddMouseFunc(std::bind(&ATBShape::MouseMotion, this, _1, _2, _3));

		XInput::XInputKeyFunc PresentGuiCanvas = [&](int key, int flags) {
			const bool isDown = (flags & 0x8000) == 0;
			const bool isRepeat = (flags & 0x4000) != 0;

			if (isDown && pxgl->GuiIsActive())
				pxgl->RenderGui(false);
			else if (isDown)
				pxgl->RenderGui(true);
		};

		pxgl->AddKeyFunc('`', PresentGuiCanvas);
		pxgl->AddKeyFunc('~', PresentGuiCanvas);

	}

	~ATBShape() {
		xprintf("ATBShape::~ATBShape()\n");
		TwTerminate();
	}

	void Draw() {
		TwDraw();
	}

	void Reshape(int w, int h) {
		TwWindowSize(w, h);
	}

	void MouseMotion(int x, int y, int f) {
		int button = (f ^ flags);
		int action = (f & 0xF)?1:0;

		if (button) {
			button--;
			TwEventMouseButtonGLFW(button, action);
		}

		TwEventMousePosGLFW(x, y);
		flags = f;
	}

	ExampleXGL *pxgl;
	int flags;
	double speed;
	bool wire;
	double time;
	glm::vec3 bgColor,cubeColor;
};

void ExampleXGL::BuildGUI() {
	AddGuiShape("shaders/000-simple", [&]() { return new ATBShape(this); });
}