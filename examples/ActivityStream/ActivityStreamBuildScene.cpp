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

namespace {
	class ActivityStreamGui : public XGLImGui 
	{
	public:
		void menuFunction() {
			if (ImGui::Begin("Activity Stream Browser", &browserWindow))
			{
				ImGui::SliderInt("Idx#", &currentViewIdx, 0, 1000);
			}
			ImGui::End();
		}

	private:
		bool browserWindow{ true };
		int currentViewIdx;
	};

	ActivityStreamGui* asGui;
}

void ExampleXGL::BuildScene() {
	XGLShape *shape;

	asGui = new ActivityStreamGui();
	menuFunctions.push_back([&]() { asGui->menuFunction(); });

	try {
		ASLink asLink;

		InitStdLog();

		std::ifstream(pathToAssets + "/assets/actor.json") >> asLink;

		std::clog << std::setw(2) << asLink << std::endl;

		using value_t = json::value_t;
		value_t type = asLink.type();

		switch (type) {
			case value_t::null:
				xprintf("type is null\n");
				break;
			
			case value_t::object:
				{
					xprintf("type is object\n");
					auto dump = asLink.dump(2);
					xprintf("Dump: {%s}\n", dump.c_str());
				}
				break;

			case value_t::array:
				xprintf("type is array\n");
				break;

			case value_t::string:
				xprintf("type is string\n");
				break;

			case value_t::boolean:
				xprintf("type is boolean\n");
				break;

			case value_t::number_integer:
				xprintf("type is number_integer\n");
				break;

			case value_t::number_unsigned:
				xprintf("type is number_unsigned\n");
				break;

			case value_t::number_float:
				xprintf("type is number_float\n");
				break;

			case value_t::discarded:
				xprintf("type is discarded\n");
				break;
		}
	}
	catch (std::exception e)
	{
		xprintf("Exception caught: %s\n", e.what());
	}

	AddShape("shaders/000-simple", [&](){ shape = new XGLTriangle(); return shape; });
}
