#include "ExampleXGL.h"
#include <AntTweakBar.h>

class ATBShape : public XGLTransformer {
public:
	ATBShape() {
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

	}

	~ATBShape() {
		xprintf("ATBShape::~ATBShape()\n");
		TwTerminate();
	}

	void Draw() {
		TwDraw();
	}

	double speed;
	bool wire;
	double time;
	glm::vec3 bgColor,cubeColor;

};

void ExampleXGL::BuildGUI() {
	AddGuiShape("shaders/000-simple", [&]() { return new ATBShape(); });

	XInputKeyFunc PresentGuiCanvas = [&](int key, int flags) {
		const bool isDown = (flags & 0x8000) == 0;
		const bool isRepeat = (flags & 0x4000) != 0;

		if (isDown && GuiIsActive())
			RenderGui(false);
		else if (isDown)
			RenderGui(true);
	};

	AddKeyFunc('`', PresentGuiCanvas);
	AddKeyFunc('~', PresentGuiCanvas);

}