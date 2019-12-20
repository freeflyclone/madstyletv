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

class ASLink : public json
{
public:
	ASLink();
};

ASLink::ASLink()
{
	emplace("@context","https://www.we.org/ns/activitystreams");
	emplace("type","Link");
}


void ExampleXGL::BuildScene() {
	XGLShape *shape;
	ASLink asLink;

	asLink["preview"] = "Your mom";
	std::cout << std::setw(4) << asLink << std::endl;

	if (asLink["preview"].size())
		xprintf("preview: %s\n", asLink["preview"].get<std::string>().c_str());

	asLink["Your Mom"] = "licked me";

	if (asLink["Your Mom"].size())
		xprintf("Your Mom: %s\n", asLink["Your Mom"].get<std::string>().c_str());

	AddShape("shaders/000-simple", [&](){ shape = new XGLTriangle(); return shape; });
}
