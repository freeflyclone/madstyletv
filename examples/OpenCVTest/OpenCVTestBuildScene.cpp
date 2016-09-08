/**************************************************************
** OpenCVTestBuildScene.cpp
**
** Just to demonstrate instantiation of a "ground"
** plane and a single triangle, with default camera manipulation
** via keyboard and mouse.
**************************************************************/
#include "ExampleXGL.h"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

void ExampleXGL::BuildScene() {
	XGLShape *shape;
	std::string imgPath = pathToAssets + "/assets/AndroidDemo.png";
	cv::Mat image;
	image = cv::imread(imgPath, cv::IMREAD_UNCHANGED);

	if (!image.data) {
		printf("imread() failed\n");
		exit(-1);
	}

	xprintf("imread() succeeded: image is %d by %d, with %d channels.\n", image.cols, image.rows, image.channels());

	//AddShape("shaders/tex", [&](){ shape = new XGLTexQuad(imgPath); return shape; });
	AddShape("shaders/tex", [&](){ shape = new XGLTexQuad(imgPath, image.cols, image.rows, image.channels(), image.data, true); return shape; });

	// have the upright texture scaled up and made 16:9 aspect, and orbiting the origin
	// to highlight use of the callback function for animation of a shape.  Note that this function
	// runs once per frame BEFORE the shape's geomentry is rendered.  A lot can
	// be done here. Hint: scripting, physics(?)
	XGLShape::AnimaFunk transform = [&](XGLShape *s, float clock) {
		float sinFunc = sin(clock / 40.0f) * 10.0f;
		float cosFunc = cos(clock / 40.0f) * 10.0f;
		glm::mat4 scale = glm::scale(glm::mat4(), glm::vec3(9.6f, 5.4f, 1.0f));
		glm::mat4 translate = glm::translate(glm::mat4(), glm::vec3(cosFunc, sinFunc, 5.4f));
		glm::mat4 rotate = glm::rotate(glm::mat4(), glm::radians(90.0f), glm::vec3(1.0f, 0.0f, 0.0f));
		s->model = translate * rotate * scale;

		s->m.ambientColor = blue;
		s->m.diffuseColor = blue;
		s->b.Bind();
	};
	//transform(shape, 0.0f);
	shape->SetTheFunk(transform);
}
