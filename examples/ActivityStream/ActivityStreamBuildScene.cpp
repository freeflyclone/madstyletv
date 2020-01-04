/**************************************************************
** ActivityStreamBuildScene.cpp
**
** Just to demonstrate instantiation of a "ground"
** plane and a single triangle, with default camera manipulation
** via keyboard and mouse.
**************************************************************/
#include <ostream>

#include "ExampleXGL.h"
#include "ActivityStream.h"
#include "DebugOutput.h"

void ExampleXGL::BuildScene() {
	XGLShape *shape;
	ASLink asLink;

	InitStdLog();

	// Create an ASLink and load it from an std::ifstream input json file
	{
		ASLink j;
		std::clog << std::setw(2) << j << std::endl;
		std::ifstream(pathToAssets + "/assets/actor.json") >> j;
		std::clog << std::setw(2) << j << std::endl;
	}

	asLink["@context"].push_back("https://e-man.tv/.well_known");
	asLink["preview"] = "https://hq.e-man.tv/hls-test.html";

	std::clog << std::setw(2) << asLink << std::endl;

	AddShape("shaders/000-simple", [&](){ shape = new XGLTriangle(); return shape; });
}
