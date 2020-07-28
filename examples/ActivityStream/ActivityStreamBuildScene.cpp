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

		// Create a temporary  ASLink and load it from an std::ifstream input json file
		{
			ASLink j;

			// First: let's log a fresh ASLink object to see its initial state.
			std::clog << "Logging 'j' immediately after creating it." << std::endl;
			std::clog << std::setw(2) << j << std::endl;

			// Second: read ActivityStream.actor object from JSON file into same ASLink object, then log that

			std::clog << "Logging 'j' with assets/actor.json update" << std::endl;
			std::ifstream(pathToAssets + "/assets/actor.json") >> j;
			std::clog << std::setw(2) << j << std::endl;
		}

		// Third: add something programatically to new ASLink object, then log that.
		asLink["@context"].push_back("https://e-man.tv/.well_known");
		asLink["preview"] = "https://hq.e-man.tv/hls-test.html";
		asLink["type"].push_back("yourMom");

		std::clog << "Logging aslink after programmatic update" << std::endl;
		std::clog << std::setw(2) << asLink << std::endl;

		std::clog << asLink["type"][0] << std::endl;
	}
	catch (std::exception e)
	{
		xprintf("Exception caught: %s\n", e.what());
	}

	AddShape("shaders/000-simple", [&](){ shape = new XGLTriangle(); return shape; });
}
