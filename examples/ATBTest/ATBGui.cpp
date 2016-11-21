#include "ExampleXGL.h"
#include <AntTweakBar.h>

class ATBShape : public XGLTransformer {
public:
	ATBShape() {
		xprintf("ATBShape::ATBShape()\n");
		TwInit(TW_OPENGL_CORE, NULL);
		TwBar *bar = TwNewBar("MadStyle TV");
	}

	~ATBShape() {
		xprintf("ATBShape::~ATBShape()\n");
		TwTerminate();
	}
};

void ExampleXGL::BuildGUI() {
	XGLShape *shape;
	AddGuiShape("shaders/000-simple", [&]() { shape = new ATBShape(); return shape; });
}