/**************************************************************
** Example05BuildScene.cpp
**
** Just to demonstrate instantiation of a "ground"
** plane and a single texture-mapped quad, along with
** scaling, translating and rotating the quad using the GLM
** functions, and doing those inside an animation callback.
**
** Overkill perhaps.
**************************************************************/
#include "ExampleXGL.h"

void ExampleXGL::BuildScene() {
	XGLGuiCanvas *shape;
	std::string imgPath = pathToAssets + "/assets/AndroidDemo.png";

	AddShape("shaders/gui-tex", [&shape,this](){
		shape = new XGLGuiCanvas(this, 1920,1080); 
		return shape; 
	});
	shape->RenderText("This is a test");
	shape->attributes.ambientColor = { 1.0, 1.0, 1.0, 0.1 };
	shape->attributes.diffuseColor = { 1.0, 1.0, 1.0, 1.0 };
	shape->model = glm::scale(glm::mat4(), glm::vec3(0.01, 0.01, 0.01));

	/*
	AddShape("shaders/tex", [&shape,imgPath](){ shape = new XGLTexQuad(imgPath); return shape; });

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
	*/
}
