/**************************************************************
** OusterDevBuildScene.cpp
**
** Let's experiment with Ouster Lidar Sensor data!
**************************************************************/
#include "ExampleXGL.h"

#include "XOuster.h"

XOuster *pOS;

void ExampleXGL::BuildScene() 
{
	try
	{
		AddShape("shaders/000-simple", [&]() {
			pOS = new XOuster(this, "C:/Users/evan/Desktop/lombard_street_OS1.raw");
			return pOS;
		});
	}
	catch (std::exception e)
	{
		xprintf("OusterSensor error: %s\n", e.what());
	}

	XInputKeyFunc frameStep = [&](int key, int flags) 
	{
		bool isDown = (flags & 0x8000) == 0;
		const bool isRepeat = (flags & 0x4000) != 0;

		if (isDown) 
		{
			if (key == 'L' || key == 'l')
			{
				pOS->StepFrame(1);
			}
			else if (key == 'H' || key == 'h')
			{
				pOS->StepFrame(-1);
			}
		}
	};

	AddKeyFunc('L', frameStep);
	AddKeyFunc('l', frameStep);
	AddKeyFunc('H', frameStep);
	AddKeyFunc('h', frameStep);

	pOS->SetAnimationFunction([&](float clock) {
		pOS->StepFrame(3);
	});
}
