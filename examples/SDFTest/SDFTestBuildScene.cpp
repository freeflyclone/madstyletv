/**************************************************************
** SDFTestBuildScene.cpp
**
** Demonstrate Signed Distance Function text rendering.
** Default camera manipulation via keyboard and mouse.
**************************************************************/
#include "ExampleXGL.h"
#include "arial.ttf_sdf.h"

struct SDF_LayoutRaw {
	int id;
	int x, y;
	int w, h;
	int xOff, yOff;
	int xAdvance;
};

struct SDF_LayoutBaked {
	int id;
	int x, y, w, h;

	float width, height, xOff, yOff, xAdvance;
};

class SDFfont {
public:
	SDFfont();
	virtual ~SDFfont();
	void Cook(int idx);

	SDF_LayoutRaw *raw;
	std::vector<SDF_LayoutBaked> baked;

	int width, height;
	int nChars;
	float scale;
};

SDFfont::SDFfont() {
	xprintf("SDFfont::SDFfont()\n");
	xprintf("sizeof layout: %d\n", sizeof(*raw));

	width = sdf_tex_width;
	height = sdf_tex_height;
	nChars = sdf_num_chars;
	scale = scale_factor;

	raw = (SDF_LayoutRaw *)sdf_spacing;

	for (int i = 0; i < nChars; i++)
		Cook(i);

	xprintf("baked has %d entries\n", baked.size());
}

SDFfont::~SDFfont() {
	xprintf("SDFfont::~SDFfont()\n");
}

void SDFfont::Cook(int i) {
	SDF_LayoutBaked b;

	b.id = raw[i].id;
	b.x = raw[i].x;
	b.y = raw[i].y;
	b.w = raw[i].w;
	b.h = raw[i].h;
	b.width = (float)b.w / scale;
	b.height = (float)b.h / scale;
	b.xOff = (float)raw[i].xOff / scale;
	b.yOff = (float)raw[i].yOff / scale;
	b.xAdvance = raw[i].xAdvance / scale;
	baked.push_back(b);

	xprintf("'%c'(%d) - w,h: %0.4f,%0.4f\n", i, i, b.width, b.height);
}

SDFfont sdfFont;

void ExampleXGL::BuildScene() {
	XGLShape *shape;
	std::string imgPath = pathToAssets + "/assets/arial.ttf_sdf.png";
	AddShape("shaders/text-sdf", [&](){ shape = new XGLTexQuad(imgPath); return shape; });

	glm::mat4 scale = glm::scale(glm::mat4(), glm::vec3(9.6f, 5.4f, 1.0f));
	glm::mat4 translate = glm::translate(glm::mat4(), glm::vec3(0.0, 0.0, 5.4f));
	glm::mat4 rotate = glm::rotate(glm::mat4(), glm::radians(90.0f), glm::vec3(1.0f, 0.0f, 0.0f));
	shape->model = translate * rotate * scale;
}
